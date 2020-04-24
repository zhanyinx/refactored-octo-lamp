import numpy as np
import scipy.ndimage as ndi
import skimage.measure
import skimage.morphology
import tensorflow as tf


def next_power(x: int, k: int = 2) -> int:
    """ Calculates x's next higher power of k. """
    y, power = 0, 1
    while y < x:
        y = k**power
        power += 1
    return y


def random_cropping(image: np.ndarray,
                    mask: np.ndarray,
                    crop_size: int = 256) -> np.ndarray:
    """
    Randomly crops an image and mask to size crop_size.
    Args:
        - image: Image to be cropped.
        - mask: Mask to be cropped.
        - crop_size: Size to crop image and mask (both dimensions).
    Returns:
        - crop_image, crop_mask: Cropped image and mask
            respectively with shape (crop_size, crop_size).
    """
    if not all(isinstance(i, np.ndarray) for i in [image, mask]):
        raise TypeError(
            f"image, mask must be np.ndarray but is {type(image), type(mask)}.")
    if not isinstance(crop_size, int):
        raise TypeError(f"crop_size must be an int but is {type(crop_size)}.")
    if not image.shape[:2] == mask.shape[:2]:
        raise ValueError(
            f"image, mask must match shape: {image.shape[:2]} != {mask.shape[:2]}.")
    if crop_size == 0:
        raise ValueError("crop_size must be larger than 0.")
    if not all(image.shape[i] >= crop_size for i in range(2)):
        raise ValueError("crop_size must be smaller than image_size.")

    start_dim = [0, 0]
    if image.shape[0] > crop_size:
        start_dim[0] = np.random.randint(
            low=0, high=image.shape[0] - crop_size)
    if image.shape[1] > crop_size:
        start_dim[1] = np.random.randint(
            low=0, high=image.shape[1] - crop_size)

    stacked_image = np.stack([image, mask])
    cropped_image = stacked_image[:, start_dim[0]:start_dim[0]+crop_size,
                                  start_dim[1]:start_dim[1]+crop_size]

    return cropped_image[0], cropped_image[1]


def add_complete_borders(mask: np.ndarray, border_size: int = 2) -> np.ndarray:
    """
    Adds all borders to labeled masks.
    Args:
        - mask: Mask with uniquely labeled objects to which borders will be added.
        - border_size: Size of border in pixels.
    Returns:
        - output_mask: Mask with three channels – Background, Objects, and Borders.
    """
    if not isinstance(mask, np.ndarray):
        raise TypeError(
            f"input_mask must be a np.ndarray but is a {type(mask)}.")
    if not isinstance(border_size, int):
        raise TypeError(f"size must be an int but is a {type(border_size)}.")

    size = int(np.ceil(border_size/2))

    borders = np.zeros(mask.shape)
    for i in np.unique(mask):
        curr_mask = np.where(mask == i, 1, 0)
        mask_dil = ndi.morphology.binary_dilation(curr_mask, iterations=size)
        mask_ero = ndi.morphology.binary_erosion(curr_mask, iterations=size)
        mask_border = np.logical_xor(mask_dil, mask_ero)
        borders[mask_border] = i
    output_mask = np.where(borders > 0, 2, mask > 0)
    output_mask = tf.keras.utils.to_categorical(output_mask)

    empty_mask = np.expand_dims(np.zeros((mask > 0).shape), axis=-1)

    # Empty mask
    if output_mask.shape[-1] == 1:
        output_mask = np.concatenate([output_mask, empty_mask], axis=-1)

    # Mask without borders
    if output_mask.shape[-1] == 2:
        output_mask = np.concatenate([output_mask, empty_mask], axis=-1)

    return output_mask


def add_touching_borders(mask: np.ndarray, border_size: int = 2) -> np.ndarray:
    """
    Adds touching borders only to labeled masks.
    Args:
        - mask: Mask with uniquely labeled objects to which borders will be added.
        - border_size: Size of border in pixels.
    Returns:
        - output_mask: Mask with three channels – Background, Objects, and Borders.
    """
    if not isinstance(mask, np.ndarray):
        raise TypeError(
            f"input_mask must be a np.ndarray but is a {type(mask)}.")
    if not isinstance(border_size, int):
        raise TypeError(f"size must be an int but is a {type(border_size)}.")

    size = int(np.ceil(border_size/2))

    borders = []
    for i in np.unique(mask)[1:]:
        curr_mask = np.where(mask == i, 1, 0)
        dilated = ndi.morphology.binary_dilation(curr_mask, iterations=size)
        eroded = ndi.morphology.binary_erosion(curr_mask, iterations=size)
        curr_border = np.logical_xor(dilated, eroded)
        borders.append(curr_border)
    borders = np.sum(borders, axis=0)
    output_mask = np.where(borders > 1, 2, mask > 0)
    output_mask = tf.keras.utils.to_categorical(output_mask)

    empty_mask = np.expand_dims(np.zeros((mask > 0).shape), axis=-1)

    # Empty mask
    if output_mask.shape[-1] == 1:
        output_mask = np.concatenate([output_mask, empty_mask], axis=-1)

    # Mask without borders
    if output_mask.shape[-1] == 2:
        output_mask = np.concatenate([output_mask, empty_mask], axis=-1)

    return output_mask


def get_centroid_diamonds(mask: np.ndarray, size: int, check_overlap: bool = False) -> np.ndarray:
    """
    Returns a image with distance transformed diamonds at the previous
    centroids of mask labels.
    Args:
        - mask: Mask with uniquely labeled objects to be replaced by diamonds.
        - size: Pixel size of the diamonds radius.
        - check_overlap: Ensures that diamonds can't overlap.
            More computationally expensive.
    Returns:
        - output_mask: Mask with diamonds in place of labeled objects.
    """
    if not isinstance(mask, np.ndarray):
        raise TypeError(
            f"input_mask must be a np.ndarray but is a {type(mask)}.")
    if not isinstance(size, int):
        raise TypeError(f"size must be an int but is a {type(size)}.")
    if not isinstance(check_overlap, bool):
        raise TypeError(
            f"check_overlap must be a bool but is a {type(check_overlap)}.")
    if size <= 0:
        raise ValueError(f"size must be a positive int but is {size}.")

    if len(np.unique(mask)) == 1:
        return mask

    diamond = skimage.morphology.diamond(size)
    diamond = ndi.distance_transform_edt(diamond)

    rprops = skimage.measure.regionprops(mask.astype(int))
    pad = size+1
    output_mask = np.zeros(mask.shape)
    output_mask = np.pad(output_mask, pad)

    if check_overlap:
        for prop in rprops:
            r, c = prop.centroid
            r, c = int(r)+pad, int(c)+pad
            curr_mask = np.zeros(mask.shape)
            curr_mask = np.pad(curr_mask, pad)
            curr_mask[r-size:r+pad, c-size:c+pad] = diamond
            output_mask = np.add(curr_mask, output_mask)
    else:
        for prop in rprops:
            r, c = prop.centroid
            r, c = int(r)+pad, int(c)+pad
            output_mask[r-size:r+pad, c-size:c+pad] = diamond

    output_mask = output_mask[pad:-pad, pad:-pad]
    return output_mask


def normalize_images(images: np.ndarray) -> np.ndarray:
    """ Normalizes images based on bit depth. """
    if images.dtype == np.uint8:
        return (images / 255).astype(np.float32)
    if images.dtype == np.uint16:
        return (images / 65535).astype(np.float32)

    return images
