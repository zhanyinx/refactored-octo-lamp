import numpy as np


def next_power(x: int, k: int = 2) -> int:
    """ Calculates x's next higher power of k. """
    y, power = 0, 1
    while y < x:
        y = k ** power
        power += 1
    return y


def random_cropping(image: np.ndarray, mask: np.ndarray, cell_size: int, crop_size: int = 256) -> np.ndarray:
    """
    Randomly crops an image and mask to size crop_size.
    Args:
        - image: Image to be cropped.
        - mask: Mask to be cropped.
        - cell_size: size of cell used to calculate F1 score
        - crop_size: Size to crop image and mask (both dimensions).
    Returns:
        - crop_image, crop_mask: Cropped image and mask
            respectively with shape (crop_size, crop_size).
    """
    if not all(isinstance(i, np.ndarray) for i in [image, mask]):
        raise TypeError(f"image, mask must be np.ndarray but is {type(image), type(mask)}.")
    if not all(isinstance(i, int) for i in [crop_size, cell_size]):
        raise TypeError(f"crop_size, cell_size must be an int but is {type(crop_size), type(cell_size)}.")
    if crop_size == 0:
        raise ValueError("crop_size must be larger than 0.")
    if not all(image.shape[i] >= crop_size for i in range(2)):
        raise ValueError("crop_size must be smaller than image_size.")
    if crop_size % cell_size > 0:
        raise ValueError("Crop size must be a multiple of cell_size.")

    start_dim = [0, 0]
    if image.shape[0] > crop_size:
        start_dim[0] = int(np.floor(np.random.randint(low=0, high=image.shape[0] - crop_size) / cell_size) * cell_size)
    if image.shape[1] > crop_size:
        start_dim[1] = int(np.floor(np.random.randint(low=0, high=image.shape[1] - crop_size) / cell_size) * cell_size)

    cropped_image = image[start_dim[0] : (start_dim[0] + crop_size), start_dim[1] : (start_dim[1] + crop_size)]
    cropped_mask = mask[
        int(start_dim[0] / cell_size) : int((start_dim[0] + crop_size) / cell_size),
        int(start_dim[1] / cell_size) : int((start_dim[1] + crop_size) / cell_size),
        :,
    ]

    return cropped_image, cropped_mask


def normalize_images(images: np.ndarray) -> np.ndarray:
    """ Normalizes images based on bit depth. """
    if images.dtype == np.uint8:
        return (images / 255).astype(np.float32)
    if images.dtype == np.uint16:
        return (images / 65535).astype(np.float32)

    return images
