import numpy as np
import scipy.ndimage as ndi
import tensorflow as tf


def next_power(x, k=2):
    """ Calculates x's next higher power of k. """
    y, power = 0, 1
    while y < x:
        y = k**power
        power += 1
    return y


def random_cropping(image: np.ndarray,
                    mask: np.ndarray,
                    crop_size: int = 256):
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
        - mask: Mask with uniquely labeled object to which borders will be added.
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

    empty_mask = np.expand_dims(np.zeros((mask>0).shape), axis=-1)

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
        - mask: Mask with uniquely labeled object to which borders will be added.
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

    empty_mask = np.expand_dims(np.zeros((mask>0).shape), axis=-1)

    # Empty mask
    if output_mask.shape[-1] == 1:
        output_mask = np.concatenate([output_mask, empty_mask], axis=-1)

    # Mask without borders
    if output_mask.shape[-1] == 2:
        output_mask = np.concatenate([output_mask, empty_mask], axis=-1)

    return output_mask
