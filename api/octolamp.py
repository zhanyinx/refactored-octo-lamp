"""Application functions."""

import logging
import os
import uuid
import glob
import sys
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

sys.path.append("../")

from api.util_image import adaptive_imread, adaptive_preprocessing, ALLOWED_EXTENSIONS, adaptive_imsave
from api.util_model import adaptive_prediction
from api.util import f1_l2_combined_loss, f1_score, l2_norm

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["SECRET_KEY"] = str(uuid.uuid4())
app.config["MODEL_FOLDER"] = "static/models"
app.config["UPLOAD_FOLDER"] = "static/tmp_upload"
app.config["DISPLAY_FOLDER"] = "static/tmp_display"
app.config["PROCESSED_FOLDER"] = "static/tmp_processed"
app.jinja_env.filters["zip"] = zip  # pylint: disable=no-member


IMAGE_TYPES = ["One Frame (Grayscale or RGB)", "Z-Stack (max projection)"]
MODEL_IDS = {
    "Ivana spots": "model_Ivana_spots_1589820419",
    "Pia spots": "model_pia_spots_1589820239",
    "Synthetic spots": "model_octo-lamp_1590569900",
}
LOCALISATION_TYPES = ["Neural network", "Gaussian fitting"]
MODELS: Dict[str, tf.keras.models.Model] = {}
LOG_FORMAT = "%(levelname)s %(asctime)s - %(filename)s %(funcName)s %(lineno)s - %(message)s"
logging.basicConfig(filename="./spotlify.log", level=logging.DEBUG, format=LOG_FORMAT, filemode="a")
log = logging.getLogger()


########################################
# Utils / helper functions
########################################
def load_model(model_id: str) -> tf.keras.models.Model:
    """Load local or downloads and loads models into memory from google drive h5 files."""
    model_file = os.path.join(app.config["MODEL_FOLDER"], f"{model_id}.h5")
    if not os.path.exists(model_file):
        model_file = gdown.download(f"https://drive.google.com/uc?id={model_id}", model_file)
    model = tf.keras.models.load_model(
        model_file, {"f1_l2_combined_loss": f1_l2_combined_loss, "f1_score": f1_score, "l2_norm": l2_norm}
    )
    return model


def predict(
    file: Dict, image_type: str, model_type: str, localisator: str = "Network localisation", single_: bool = False
) -> Iterable[str]:
    """Adaptively preprocesses, predicts, and saves images returning the filename(s)."""
    log.info(f'Predicting with file "{file}", image "{image_type}", model {model_type}".')

    # Get filename from request.form.get
    fname = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(file.filename))  # type: ignore[attr-defined]

    # Output format
    ext = "tiff"
    if single_:
        ext = "png"

    # Naming
    base = f'{secure_filename(file.filename).split(".")[0]}'  # type: ignore[attr-defined]
    basename = f'{base}.{ext}'
    fname_in = os.path.join(app.config["DISPLAY_FOLDER"], basename)
    fname_out_image = os.path.join(app.config["PROCESSED_FOLDER"], basename)
    file.save(fname)  # type: ignore[attr-defined]
    log.info(f'File "{fname}" saved.')

    # Processing
    original = adaptive_imread(file=fname)
    preprocessed = adaptive_preprocessing(image=original, image_type=image_type)
    prediction = adaptive_prediction(images=preprocessed, model=MODELS[model_type], localisator=localisator)
    log.info(f"Prediction returned.")

    # Saving
    if single_:
        adaptive_imsave(fname=fname_in, image=preprocessed, image_type=image_type)

    adaptive_imsave(fname=fname_out_image, image=preprocessed, image_type=image_type, prediction=prediction)
    log.info(f"Predictions image saved.")

    fname_out_coordinates = os.path.join(app.config["PROCESSED_FOLDER"], f'{base}.csv')
    prediction.to_csv(fname_out_coordinates, index=False)
    log.info(f"Predictions coordinates saved.")

    if single_:
        return fname_in, fname_out_image, fname_out_coordinates
    return fname_out_coordinates


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
    localisation_selection = request.cookies.get("localisation_selection")

    return render_template(
        "single.html",
        title="Single Images",
        image_options=IMAGE_TYPES,
        image_selection=image_selection,
        model_options=list(MODEL_IDS.keys()),
        model_selection=model_selection,
        localisation_options=LOCALISATION_TYPES,
        localisation_selection=localisation_selection,
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

    fname_in, fname_out_image, fname_out_coordinates = predict(
        file=file, image_type=image_selection, model_type=model_selection, localisator=loc, single_=True
    )

    pred_display = fname_out_image

    resp = make_response(
        render_template(
            "prediction.html",
            title="Prediction",
            original=fname_in,
            prediction=fname_out_image,
            prediction_coord=fname_out_coordinates,
            prediction_display=pred_display,
        )
    )
    resp.set_cookie("image_selection", image_selection)
    resp.set_cookie("model_selection", model_selection)
    resp.set_cookie("localisation_selection", loc)
    return resp


########################################
# Batch predictions
########################################


@app.route("/batch")
def batch():
    """Standard "Batch Processing" page."""
    image_selection = request.cookies.get("image_selection")
    model_selection = request.cookies.get("model_selection")
    localisation_selection = request.cookies.get("localisation_selection")
    image_types = IMAGE_TYPES.copy()
    image_types.append("All Frames")
    image_types.append("Time-Series")

    return render_template(
        "batch.html",
        title="Batch Prediction",
        image_options=image_types,
        image_selection=image_selection,
        model_options=list(MODEL_IDS.keys()),
        model_selection=model_selection,
        localisation_options=LOCALISATION_TYPES,
        localisation_selection=localisation_selection,
    )


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    """API to predict on multiple image files, returning the location of predicted images."""
    uuids = request.form.getlist("uuid")
    files = request.files.getlist("file")
    image_selection = request.form.get("image")
    model_selection = request.form.get("model")
    loc = request.form.get("localisation")
    log.info(
        f"Batch selections - ids: {uuids}, files: {files}, image: {image_selection}, \
        model: {model_selection}, localisation: {loc}."
    )

    response = {uuid: predict(file, image_selection, model_selection, loc) for uuid, file in zip(uuids, files)}
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
