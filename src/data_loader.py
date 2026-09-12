from pathlib import Path
import json
import numpy as np


CLASSES = [
    "canola",
    "kochia",
    "sugarbeet",
    "redroot_pigweed",
    "ragweed",
    "waterhemp",
    "soybean",
]


def load_split(split_name):
    """
    Load one dataset split.

    Parameters
    ----------
    split_name : str
        One of: train, val, test

    Returns
    -------
    X : numpy.ndarray
        Hyperspectral cubes with shape (N, 100, 100, 54)

    y : numpy.ndarray
        Integer class labels with shape (N,)
    """

    split_file = Path("data/splits") / f"{split_name}.json"

    if not split_file.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_file}"
        )

    with open(split_file, "r") as f:
        records = json.load(f)

    X = []
    y = []

    print(f"\nLoading {split_name} split...")
    print(f"Number of samples: {len(records)}")

    for i, record in enumerate(records, start=1):

        file_path = Path(record["file"])

        if not file_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {file_path}"
            )

        cube = np.load(file_path)

        if cube.shape != (100, 100, 54):
            raise ValueError(
                f"Unexpected shape in {file_path}: {cube.shape}"
            )

        X.append(cube)
        y.append(record["label"])

    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)

    print(f"Loaded X shape: {X.shape}")
    print(f"Loaded y shape: {y.shape}")

    return X, y


def get_class_names():
    """Return the class names in label order."""
    return CLASSES


if __name__ == "__main__":

    print("=" * 60)
    print("Testing 7-class WeedCube data loader")
    print("=" * 60)

    for split in ["train", "val", "test"]:
        X, y = load_split(split)

        print(f"\n{split.upper()}:")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        print(f"Labels: {np.unique(y)}")

    print("\nClass mapping:")

    for i, class_name in enumerate(CLASSES):
        print(f"{i}: {class_name}")

    print("\nData loader test completed successfully!")