import tensorflow as tf
from tensorflow.keras import layers, regularizers


L2_REG = 0.0001
DROPOUT_RATE = 0.5
NUM_CLASSES = 7


def residual_block(x, filters, stride=1):
    shortcut = x

    # First convolution
    x = layers.Conv3D(
        filters,
        kernel_size=(3, 3, 3),
        strides=(stride, stride, stride),
        padding="same",
        use_bias=False,
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Second convolution
    x = layers.Conv3D(
        filters,
        kernel_size=(3, 3, 3),
        strides=(1, 1, 1),
        padding="same",
        use_bias=False,
    )(x)

    x = layers.BatchNormalization()(x)

    # Adjustment layer when dimensions change
    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = layers.Conv3D(
            filters,
            kernel_size=(1, 1, 1),
            strides=(stride, stride, stride),
            padding="same",
            use_bias=False,
        )(shortcut)

        shortcut = layers.BatchNormalization()(shortcut)

    # Residual connection
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)

    return x


def build_hr_cnn(input_shape=(100, 100, 54, 1), num_classes=NUM_CLASSES):

    inputs = layers.Input(shape=input_shape)

    # Initial convolution
    x = layers.Conv3D(
        32,
        kernel_size=(3, 3, 3),
        strides=(2, 2, 2),
        padding="same",
        use_bias=False,
    )(inputs)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # ---------------------------------------------------------
    # Residual stage 1: 32 filters
    # ---------------------------------------------------------
    x = residual_block(x, 32, stride=1)
    x = residual_block(x, 32, stride=1)

    # ---------------------------------------------------------
    # Residual stage 2: 64 filters
    # ---------------------------------------------------------
    x = residual_block(x, 64, stride=2)
    x = residual_block(x, 64, stride=1)

    # ---------------------------------------------------------
    # Residual stage 3: 128 filters
    # ---------------------------------------------------------
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128, stride=1)

    # ---------------------------------------------------------
    # Residual stage 4: 256 filters
    # ---------------------------------------------------------
    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 256, stride=1)

    # ---------------------------------------------------------
    # Classification head
    # Paper describes max pooling -> flatten -> dense layers
    # ---------------------------------------------------------
    x = layers.MaxPooling3D(
        pool_size=(2, 2, 2),
        strides=(2, 2, 2),
        padding="same",
    )(x)

    x = layers.Flatten()(x)

    x = layers.Dense(
        512,
        activation="relu",
        kernel_regularizer=regularizers.l2(L2_REG),
    )(x)

    x = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=regularizers.l2(L2_REG),
    )(x)

    x = layers.Dropout(DROPOUT_RATE)(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
    )(x)

    return tf.keras.Model(
        inputs,
        outputs,
        name="HR-CNN",
    )


if __name__ == "__main__":

    print("=" * 60)
    print("Testing HR-CNN architecture")
    print("=" * 60)

    model = build_hr_cnn()

    print("\nModel created successfully!")
    print(f"Input shape : {model.input_shape}")
    print(f"Output shape: {model.output_shape}")

    print("\nModel summary:\n")
    model.summary()