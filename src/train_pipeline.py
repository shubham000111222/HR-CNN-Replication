import sys
from pathlib import Path

import numpy as np

# Allow importing modules from src/
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from data_loader import load_split
from augmentation import augment_dataset


def prepare_data():
    """
    Prepare training, validation, and test data.

    Augmentation is applied ONLY to the training set.
    """

    print("=" * 60)
    print("Preparing HR-CNN training pipeline")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Load original splits
    # ---------------------------------------------------------

    print("\n[1] Loading training data...")
    X_train, y_train = load_split("train")

    print("\n[2] Loading validation data...")
    X_val, y_val = load_split("val")

    print("\n[3] Loading test data...")
    X_test, y_test = load_split("test")

    # ---------------------------------------------------------
    # 2. Augment TRAINING data only
    # ---------------------------------------------------------

    print("\n[4] Augmenting training data...")

    X_train_aug, y_train_aug = augment_dataset(
        X_train,
        y_train
    )

    print(f"Original training samples : {len(X_train)}")
    print(f"Augmented training samples: {len(X_train_aug)}")

    # ---------------------------------------------------------
    # 3. Add channel dimension for Conv3D
    # ---------------------------------------------------------
    #
    # Current:
    #     (N, 100, 100, 54)
    #
    # Required by Conv3D:
    #     (N, 100, 100, 54, 1)
    #
    # ---------------------------------------------------------

    X_train_aug = np.expand_dims(X_train_aug, axis=-1)
    X_val = np.expand_dims(X_val, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    # ---------------------------------------------------------
    # 4. Print final shapes
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL DATA SHAPES")
    print("=" * 60)

    print(f"Training   X: {X_train_aug.shape}")
    print(f"Training   y: {y_train_aug.shape}")

    print(f"Validation X: {X_val.shape}")
    print(f"Validation y: {y_val.shape}")

    print(f"Test       X: {X_test.shape}")
    print(f"Test       y: {y_test.shape}")

    # ---------------------------------------------------------
    # 5. Check labels
    # ---------------------------------------------------------

    print("\nLabel verification:")

    print("Training labels  :", np.unique(y_train_aug))
    print("Validation labels:", np.unique(y_val))
    print("Test labels      :", np.unique(y_test))

    # ---------------------------------------------------------
    # 6. Check numerical values
    # ---------------------------------------------------------

    print("\nValue ranges:")

    print(
        f"Training   : "
        f"{X_train_aug.min():.6f} → {X_train_aug.max():.6f}"
    )

    print(
        f"Validation : "
        f"{X_val.min():.6f} → {X_val.max():.6f}"
    )

    print(
        f"Test       : "
        f"{X_test.min():.6f} → {X_test.max():.6f}"
    )

    print("\n" + "=" * 60)
    print("Training pipeline test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    prepare_data()