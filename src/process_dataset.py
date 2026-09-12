from pathlib import Path
import argparse
import numpy as np

from preprocessing import preprocess_cube


def process_class(class_name):
    """
    Process all hyperspectral .npy files for one WeedCube class.
    """

    input_dir = Path("data/raw/weedcube") / class_name / class_name
    output_dir = Path("data/processed") / class_name

    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.npy"))

    if not files:
        raise FileNotFoundError(
            f"No .npy files found in: {input_dir}"
        )

    print(f"Class: {class_name}")
    print(f"Found {len(files)} hyperspectral files.")

    for i, file_path in enumerate(files, start=1):

        output_path = output_dir / file_path.name

        print(f"\n[{i}/{len(files)}] Processing {file_path.name}")

        # Load one cube at a time
        cube = np.load(file_path)

        print(f"  Original shape: {cube.shape}")

        # 224 -> 54 channels and resize to 100 x 100
        processed = preprocess_cube(cube)

        print(f"  Processed shape: {processed.shape}")

        np.save(output_path, processed)

        print(f"  Saved: {output_path}")

        del cube
        del processed

    print(f"\nAll {class_name} files processed successfully!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "class_name",
        help="Class name, e.g. kochia or canola"
    )

    args = parser.parse_args()

    process_class(args.class_name)