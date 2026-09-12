from pathlib import Path
import random
import json

# Reproducibility
SEED = 42
random.seed(SEED)

# Dataset location
DATA_DIR = Path("data/processed")
SPLIT_DIR = Path("data/splits")
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

# Public WeedCube classes
CLASSES = [
    "canola",
    "kochia",
    "sugarbeet",
    "redroot_pigweed",
    "ragweed",
    "waterhemp",
    "soybean",
]

# 6:2:2 split
TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
TEST_RATIO = 0.20


def create_split():
    splits = {
        "train": [],
        "val": [],
        "test": [],
    }

    print("Creating stratified 6:2:2 split...\n")

    for class_name in CLASSES:
        class_dir = DATA_DIR / class_name
        files = sorted(class_dir.glob("*.npy"))

        if not files:
            raise FileNotFoundError(
                f"No processed files found for class: {class_name}"
            )

        # Shuffle deterministically
        random.shuffle(files)

        n = len(files)

        train_end = int(n * TRAIN_RATIO)
        val_end = train_end + int(n * VAL_RATIO)

        train_files = files[:train_end]
        val_files = files[train_end:val_end]
        test_files = files[val_end:]

        for file_path in train_files:
            splits["train"].append({
                "file": str(file_path),
                "class": class_name,
                "label": CLASSES.index(class_name),
            })

        for file_path in val_files:
            splits["val"].append({
                "file": str(file_path),
                "class": class_name,
                "label": CLASSES.index(class_name),
            })

        for file_path in test_files:
            splits["test"].append({
                "file": str(file_path),
                "class": class_name,
                "label": CLASSES.index(class_name),
            })

        print(
            f"{class_name:20s} "
            f"Total={n:2d}  "
            f"Train={len(train_files):2d}  "
            f"Val={len(val_files):2d}  "
            f"Test={len(test_files):2d}"
        )

    # Save splits
    for split_name, data in splits.items():
        output_file = SPLIT_DIR / f"{split_name}.json"

        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        print(f"\nSaved: {output_file}")

    print("\n" + "=" * 60)
    print("Split complete!")
    print("=" * 60)

    for split_name, data in splits.items():
        print(f"{split_name.capitalize():5s}: {len(data)} images")


if __name__ == "__main__":
    create_split()