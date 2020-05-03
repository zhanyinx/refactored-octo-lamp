import numpy as np


def next_power(x: int, k: int = 2) -> int:
    """ Calculates x's next higher power of k. """
    y, power = 0, 1
    while y < x:
        y = k ** power
        power += 1
    return y


def random_cropping(image: np.ndarray, mask: np.ndarray, crop_size: int = 256) -> np.ndarray:
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
        raise TypeError(f"image, mask must be np.ndarray but is {type(image), type(mask)}.")
    if not isinstance(crop_size, int):
        raise TypeError(f"crop_size must be an int but is {type(crop_size)}.")
    if not image.shape[:2] == mask.shape[:2]:
        raise ValueError(f"image, mask must match shape: {image.shape[:2]} != {mask.shape[:2]}.")
    if crop_size == 0:
        raise ValueError("crop_size must be larger than 0.")
    if not all(image.shape[i] >= crop_size for i in range(2)):
        raise ValueError("crop_size must be smaller than image_size.")

    start_dim = [0, 0]
    if image.shape[0] > crop_size:
        start_dim[0] = np.random.randint(low=0, high=image.shape[0] - crop_size)
    if image.shape[1] > crop_size:
        start_dim[1] = np.random.randint(low=0, high=image.shape[1] - crop_size)

    stacked_image = np.stack([image, mask])
    cropped_image = stacked_image[:, start_dim[0] : start_dim[0] + crop_size, start_dim[1] : start_dim[1] + crop_size]

    return cropped_image[0], cropped_image[1]


def normalize_images(images: np.ndarray) -> np.ndarray:
    """ Normalizes images based on bit depth. """
    if images.dtype == np.uint8:
        return (images / 255).astype(np.float32)
    if images.dtype == np.uint16:
        return (images / 65535).astype(np.float32)

    return images
