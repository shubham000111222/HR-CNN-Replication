import numpy as np
from skimage.transform import resize


def reduce_spectral_channels(cube):
    """
    Reduce a hyperspectral cube from 224 channels to 54 channels.
    """
    if cube.ndim != 3:
        raise ValueError(
            f"Expected a 3D cube (height, width, channels), got {cube.shape}"
        )

    if cube.shape[-1] != 224:
        raise ValueError(
            f"Expected 224 spectral channels, got {cube.shape[-1]}"
        )

    return cube[:, :, 62:-1:3]


def resize_cube(cube, size=(100, 100)):
    """
    Resize spatial dimensions while keeping the spectral dimension.
    """
    height, width, channels = cube.shape

    resized = resize(
        cube,
        (size[0], size[1], channels),
        preserve_range=True,
        anti_aliasing=True,
    )

    return resized.astype(np.float32)


def preprocess_cube(cube, size=(100, 100)):
    """
    Complete preprocessing:
    224 channels -> 54 channels
    H x W -> 100 x 100
    """
    cube = reduce_spectral_channels(cube)
    cube = resize_cube(cube, size)

    return cube