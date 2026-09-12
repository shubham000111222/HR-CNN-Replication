import numpy as np


def rotate_cube(cube, k):
    """
    Rotate hyperspectral cube by 90-degree increments.

    k=1 -> 90 degrees
    k=2 -> 180 degrees
    k=3 -> 270 degrees
    """
    return np.rot90(cube, k=k, axes=(0, 1)).copy()


def horizontal_flip(cube):
    """Flip the cube horizontally."""
    return np.flip(cube, axis=1).copy()


def vertical_flip(cube):
    """Flip the cube vertically."""
    return np.flip(cube, axis=0).copy()


def channel_flip(cube):
    """
    Reverse the spectral-channel order.
    """
    return np.flip(cube, axis=2).copy()


def translate_cube(cube, shift_x, shift_y):
    """
    Translate the hyperspectral cube.

    Empty areas are filled with zeros.

    shift_x > 0 -> shift right
    shift_x < 0 -> shift left
    shift_y > 0 -> shift down
    shift_y < 0 -> shift up
    """

    height, width, channels = cube.shape

    translated = np.zeros_like(cube)

    # Source coordinates
    src_x_start = max(0, -shift_x)
    src_x_end = min(width, width - shift_x)

    src_y_start = max(0, -shift_y)
    src_y_end = min(height, height - shift_y)

    # Destination coordinates
    dst_x_start = max(0, shift_x)
    dst_x_end = dst_x_start + (src_x_end - src_x_start)

    dst_y_start = max(0, shift_y)
    dst_y_end = dst_y_start + (src_y_end - src_y_start)

    translated[
        dst_y_start:dst_y_end,
        dst_x_start:dst_x_end,
        :
    ] = cube[
        src_y_start:src_y_end,
        src_x_start:src_x_end,
        :
    ]

    return translated


def augment_cube(cube):
    """
    Generate augmented versions of one hyperspectral cube.

    Returns a list containing the original cube and
    augmented versions.
    """

    augmented = []

    # Original
    augmented.append(cube.copy())

    # Rotations
    augmented.append(rotate_cube(cube, 1))
    augmented.append(rotate_cube(cube, 2))
    augmented.append(rotate_cube(cube, 3))

    # Spatial flips
    augmented.append(horizontal_flip(cube))
    augmented.append(vertical_flip(cube))

    # Spectral/channel flip
    augmented.append(channel_flip(cube))

    # Translations
    augmented.append(translate_cube(cube, 10, 0))
    augmented.append(translate_cube(cube, -10, 0))
    augmented.append(translate_cube(cube, 0, 10))
    augmented.append(translate_cube(cube, 0, -10))

    return augmented


def augment_dataset(X, y):
    """
    Augment an entire training dataset.

    Parameters
    ----------
    X : numpy.ndarray
        Shape: (N, 100, 100, 54)

    y : numpy.ndarray
        Shape: (N,)

    Returns
    -------
    X_aug : numpy.ndarray
        Augmented images.

    y_aug : numpy.ndarray
        Corresponding labels.
    """

    X_augmented = []
    y_augmented = []

    for cube, label in zip(X, y):

        augmented_cubes = augment_cube(cube)

        for aug_cube in augmented_cubes:
            X_augmented.append(aug_cube)
            y_augmented.append(label)

    X_augmented = np.stack(X_augmented).astype(np.float32)
    y_augmented = np.array(y_augmented, dtype=np.int64)

    return X_augmented, y_augmented


if __name__ == "__main__":

    print("=" * 60)
    print("Testing hyperspectral augmentation")
    print("=" * 60)

    # Create a dummy hyperspectral cube
    cube = np.random.rand(100, 100, 54).astype(np.float32)

    print(f"\nOriginal shape: {cube.shape}")

    augmented = augment_cube(cube)

    print(f"Number of augmented versions: {len(augmented)}")

    for i, item in enumerate(augmented):
        print(f"Augmented {i + 1}: {item.shape}")

    print("\nAugmentation test completed successfully!")