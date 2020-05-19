"""Application functions."""

import logging
import os
import uuid
import glob
from typing import Iterable, Dict

import gdown
from flask import (
    Flask,
    render_template,
    request,
    make_response,
    send_from_directory,
    jsonify,
    redirect,
    url_for,
)
from werkzeug.utils import secure_filename
import tensorflow as tf

from util_image import adaptive_imread, adaptive_preprocessing, ALLOWED_EXTENSIONS, adaptive_imsave
from util_model import adaptive_prediction

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["SECRET_KEY"] = str(uuid.uuid4())
app.config["MODEL_FOLDER"] = "static/models"
app.config["UPLOAD_FOLDER"] = "static/tmp_upload"
app.config["DISPLAY_FOLDER"] = "static/tmp_display"
app.config["PROCESSED_FOLDER"] = "static/tmp_processed"
app.jinja_env.filters["zip"] = zip  # pylint: disable=no-member


IMAGE_TYPES = ["One Frame (Grayscale or RGB)", "Z-Stack", "Time-Series"]
MODEL_IDS = {
    "Nuclear Semantic": "1XePQvBqgVx1zZZeYEryFd56ujgZumA9F",
    "Nuclear Instances": "166rnQYPQmzewIAjrbU7Ye-BhFotq2wnA",
    "Stress-Granules": "1SjjG4FbU9VkKTlN0Gvl7AsfiaoBKF7DB",
    "Cytoplasm (SunTag Background)": "1pVhyF81T4t7hh16dYKT3uJa5WCUQUI6X",
}
MODELS = {}
LOG_FORMAT = "%(levelname)s %(asctime)s - %(filename)s %(funcName)s %(lineno)s - %(message)s"
logging.basicConfig(filename="./fluffy.log", level=logging.DEBUG, format=LOG_FORMAT, filemode="a")
log = logging.getLogger()


########################################
# Utils / helper functions
########################################
def load_model(model_id: str) -> tf.keras.models.Model:
    """Downloads and loads models into memory from google drive h5 files."""
    model_file = os.path.join(app.config["MODEL_FOLDER"], f"{model_id}.h5")
    if not os.path.exists(model_file):
        model_file = gdown.download(f"https://drive.google.com/uc?id={model_id}", model_file)
    model = tf.keras.models.load_model(model_file)
    return model


def predict(
    file: Dict, image_type: str, model_type: str, localisator: str = "Network localisation", single_: bool = False
) -> Iterable[str]:
    """Adaptively preprocesses, predicts, and saves images returning the filename(s)."""
    log.info(f'Predicting with file "{file}", image "{image_type}", model {model_type}".')

    # Naming
    fname = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(file.filename))

    # Output format
    ext = "tiff"
    if single_:
        ext = "png"

    basename = f'{secure_filename(file.filename).split(".")[0]}.{ext}'
    fname_in = os.path.join(app.config["DISPLAY_FOLDER"], basename)
    fname_out = os.path.join(app.config["PROCESSED_FOLDER"], basename)
    file.save(fname)
    log.info(f'File "{fname}" saved.')

    # Processing
    original = adaptive_imread(file=fname)
    preprocessed = adaptive_preprocessing(image=original, image_type=image_type)
    prediction = adaptive_prediction(image=preprocessed, model=MODELS[model_type], localisator=localisator)
    log.info(f"Prediction returned.")

    # Saving
    if single_:
        adaptive_imsave(fname=fname_in, image=preprocessed, image_type=image_type)

    adaptive_imsave(fname=fname_out, image=prediction, image_type=image_type)
    log.info(f"Predictions saved.")

    if single_:
        return fname_in, fname_out
    return fname_out


########################################
# Setup / index
########################################
@app.before_first_request
def run_setup():
    """Setup before responding to requests. Downloads all models."""
    for name, model_id in MODEL_IDS.items():
        MODELS[name] = load_model(model_id)
        log.info(f'Loaded model "{name}".')
    log.info("Model loading complete.")


@app.route("/")
@app.route("/index")
def index():
    """Index function."""
    return render_template("index.html", landing=True)


########################################
# Single predictions
########################################


@app.route("/single")
def single():
    """Standard "Single Images" page."""
    image_selection = request.cookies.get("image_selection")
    model_selection = request.cookies.get("model_selection")

    return render_template(
        "single.html",
        title="Single Images",
        image_options=IMAGE_TYPES,
        image_selection=image_selection,
        model_options=list(MODEL_IDS.keys()),
        model_selection=model_selection,
    )


@app.route("/predict_single", methods=["POST"])
def predict_single():
    """API to predict on multiple image files, returning the location of input and predicted images."""
    file = request.files["file"]
    image_selection = request.form.get("image")
    model_selection = request.form.get("model")
    loc = request.form.get("localisation")
    log.info(
        f"Single selections - file: {file}, image: {image_selection}, model: {model_selection}, localisation: {loc}."
    )

    fname_in, fname_out, fname_instances = predict(
        file=file, image_type=image_selection, model_type=model_selection, localisator=loc, single_=True
    )
    if fname_instances is None:
        pred_display = fname_out
    else:
        pred_display = fname_instances

    resp = make_response(
        render_template(
            "prediction.html",
            title="Prediction",
            original=fname_in,
            prediction=fname_out,
            prediction_display=pred_display,
            # import itertools
            # zipped_list=list(itertools.chain(*list(zip(fname_in, fname_out))))
        )
    )
    resp.set_cookie("image_selection", image_selection)
    resp.set_cookie("model_selection", model_selection)
    return resp


########################################
# Batch predictions
########################################


@app.route("/batch")
def batch():
    """Standard "Batch Processing" page."""
    image_selection = request.cookies.get("image_selection")
    model_selection = request.cookies.get("model_selection")
    image_types = IMAGE_TYPES.copy()
    image_types.append("All Frames")

    return render_template(
        "batch.html",
        title="Batch Prediction",
        image_options=image_types,
        image_selection=image_selection,
        model_options=list(MODEL_IDS.keys()),
        model_selection=model_selection,
    )


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    """API to predict on multiple image files, returning the location of predicted images."""
    uuids = request.form.getlist("uuid")
    files = request.files.getlist("file")
    image_selection = request.form.get("image")
    model_selection = request.form.get("model")
    log.info(f"Batch selections - ids: {uuids}, files: {files}, image: {image_selection}, model: {model_selection}.")

    response = {uuid: predict(file, image_selection, model_selection) for uuid, file in zip(uuids, files)}
    return jsonify(response)


########################################
# Help
########################################


@app.route("/help")
def help():  # pylint: disable=redefined-builtin
    """Help function."""
    return render_template("help.html", title="Help")


########################################
# Utils / helper routes
########################################


@app.route("/download/<path:filename>")
def download(filename):
    """API to download a single filename."""
    filename = filename.split("/")[-1]
    return send_from_directory(app.config["PROCESSED_FOLDER"], filename, as_attachment=True)


@app.route("/delete/<path:filename>/")
def delete(filename):
    """Deletes a single filename from all temporary image folders."""
    for path in [
        "UPLOAD_FOLDER",
        "DISPLAY_FOLDER",
        "PROCESSED_FOLDER",
    ]:
        for ext in ALLOWED_EXTENSIONS:
            name = os.path.join(app.config[path], f'{filename.split(".")[0]}.{ext}')
            if os.path.exists(name):
                os.remove(name)
    return "success"


@app.route("/delete_all/<url>")
def delete_all(url):
    """Deletes all files stored in temporary image folders."""
    for path in [
        "UPLOAD_FOLDER",
        "DISPLAY_FOLDER",
        "PROCESSED_FOLDER",
    ]:
        files = glob.glob(f"{app.config[path]}/*")
        for f in files:
            os.remove(f)
    log.info("All files removed.")

    return redirect(url_for(url))


# TO DO: remember to remove debug=True in deployment
if __name__ == "__main__":
    app.run(debug=True)