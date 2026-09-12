import tensorflow as tf
from pathlib import Path

from data_loader import load_split
from augmentation import augment_cube
from models.hr_cnn import build_hr_cnn


# ============================================================
# Configuration
# ============================================================

NUM_CLASSES = 7
INPUT_SHAPE = (100, 100, 54, 1)

BATCH_SIZE = 2
EPOCHS = 400
LEARNING_RATE = 0.0001

MODEL_DIR = Path("results/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Augmentation
# ============================================================

def create_augmented_training_data(X, y):
    """
    Create augmented training data only.

    Each original cube produces 11 versions:
    original + rotations + flips + translations.
    """

    X_augmented = []
    y_augmented = []

    for cube, label in zip(X, y):

        augmented_cubes = augment_cube(cube)

        for augmented_cube in augmented_cubes:
            X_augmented.append(augmented_cube)
            y_augmented.append(label)

    X_augmented = tf.convert_to_tensor(
        X_augmented,
        dtype=tf.float32
    )

    y_augmented = tf.keras.utils.to_categorical(
        y_augmented,
        NUM_CLASSES
    )

    return X_augmented, y_augmented


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("HR-CNN TRAINING")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Load original data
    # --------------------------------------------------------

    print("\n[1] Loading training data...")
    X_train, y_train = load_split("train")

    print("\n[2] Loading validation data...")
    X_val, y_val = load_split("val")

    print("\n[3] Loading test data...")
    X_test, y_test = load_split("test")

    # --------------------------------------------------------
    # 2. Augment training data
    # --------------------------------------------------------

    print("\n[4] Augmenting training data...")

    print(f"Original training samples : {len(X_train)}")

    X_train, y_train = create_augmented_training_data(
        X_train,
        y_train
    )

    print(f"Augmented training samples: {len(X_train)}")

    # Add channel dimension for Conv3D
    X_train = X_train[..., None]
    X_val = X_val[..., None]
    X_test = X_test[..., None]

    print("\nFinal data shapes:")
    print(f"Training   X: {X_train.shape}")
    print(f"Training   y: {y_train.shape}")
    print(f"Validation X: {X_val.shape}")
    print(f"Validation y: {y_val.shape}")
    print(f"Test       X: {X_test.shape}")
    print(f"Test       y: {y_test.shape}")

    # --------------------------------------------------------
    # 3. Convert validation/test labels to one-hot
    # --------------------------------------------------------

    print("\n[5] Converting validation/test labels...")

    y_val = tf.keras.utils.to_categorical(
        y_val,
        NUM_CLASSES
    )

    y_test = tf.keras.utils.to_categorical(
        y_test,
        NUM_CLASSES
    )

    # --------------------------------------------------------
    # 4. Create TensorFlow datasets
    # --------------------------------------------------------

    print("\n[6] Creating TensorFlow datasets...")

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (X_train, y_train)
    )

    train_dataset = train_dataset.shuffle(
        buffer_size=len(X_train),
        seed=42,
        reshuffle_each_iteration=True
    )

    train_dataset = train_dataset.batch(
        BATCH_SIZE
    ).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices(
        (X_val, y_val)
    )

    val_dataset = val_dataset.batch(
        BATCH_SIZE
    ).prefetch(tf.data.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices(
        (X_test, y_test)
    )

    test_dataset = test_dataset.batch(
        BATCH_SIZE
    ).prefetch(tf.data.AUTOTUNE)

    print("Datasets created successfully.")

    # --------------------------------------------------------
    # 5. Build model
    # --------------------------------------------------------

    print("\n[7] Building HR-CNN...")

    model = build_hr_cnn(
        input_shape=INPUT_SHAPE,
        num_classes=NUM_CLASSES
    )

    print("HR-CNN created successfully.")

    # --------------------------------------------------------
    # 6. Compile
    # --------------------------------------------------------

    print("\n[8] Compiling model...")

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    )

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("Model compiled successfully.")

    # --------------------------------------------------------
    # 7. Callbacks
    # --------------------------------------------------------

    checkpoint_path = MODEL_DIR / "hr_cnn_best.keras"

    callbacks = [

        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=20,
            mode="max",
            restore_best_weights=True,
            verbose=1
        ),

        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        ),

    ]

    # --------------------------------------------------------
    # 8. Training
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    print(f"Batch size     : {BATCH_SIZE}")
    print(f"Maximum epochs : {EPOCHS}")
    print(f"Learning rate  : {LEARNING_RATE}")
    print("Optimizer      : Adam")
    print("Loss           : Categorical Cross-Entropy")
    print("Early stopping : patience = 20")
    print("=" * 60)

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # --------------------------------------------------------
    # 9. Evaluate
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("EVALUATING ON TEST SET")
    print("=" * 60)

    test_loss, test_accuracy = model.evaluate(
        test_dataset,
        verbose=1
    )

    print(f"\nTest Loss     : {test_loss:.4f}")
    print(f"Test Accuracy : {test_accuracy:.4f}")

    # --------------------------------------------------------
    # 10. Save final model
    # --------------------------------------------------------

    final_model_path = MODEL_DIR / "hr_cnn_final.keras"

    model.save(final_model_path)

    print("\nFinal model saved to:")
    print(final_model_path)

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()