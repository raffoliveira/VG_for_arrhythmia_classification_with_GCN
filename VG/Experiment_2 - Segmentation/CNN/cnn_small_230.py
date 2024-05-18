import os
from typing import Type
from tensorflow import keras
from tensorflow.keras import layers
from helpers.aux_cnn import CNNFunctions


def building_model() -> Type[keras.Model]:
    """Build the CNN model

    Returns:
        Type[keras.Model]: The built model
    """

    input_layer = keras.Input(shape=(300, 1))

    x = layers.Conv1D(filters=32, kernel_size=3, strides=2, activation="relu", padding="same")(input_layer)
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(filters=64, kernel_size=3, strides=2, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dropout(0.2)(x)

    x = layers.Flatten()(x)

    x = layers.Dense(4096, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(2048, activation="relu", kernel_regularizer=keras.regularizers.L2())(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(1024, activation="relu", kernel_regularizer=keras.regularizers.L2())(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.L2())(x)

    output_layer = layers.Dense(3, activation="softmax")(x)

    return keras.Model(inputs=input_layer, outputs=output_layer)


if __name__ == "__main__":

    cnn = CNNFunctions()
    PATH = "../../../Data"
    files = os.listdir(os.path.join(PATH, "Train"))
    files_validation = ["109.mat", "114.mat", "207.mat", "223.mat"]
    files_training = list(set(files) - set(files_validation))

    print("segmentating...")
    training_signals = cnn.segmentation_signals(
        path=PATH,
        list_ecgs=files_training,
        size_beat_before=100,
        size_beat_after=130,
        set_name="Train",
    )
    validation_signals = cnn.segmentation_signals(
        path=PATH,
        list_ecgs=files_training,
        size_beat_before=100,
        size_beat_after=130,
        set_name="Train",
    )

    print("sampling...")
    training_signals = cnn.sampling_windows_beats(signals=training_signals)
    validation_signals = cnn.sampling_windows_beats(signals=validation_signals)

    print("scaling")
    X_training, y_training, classes_weights = cnn.scaling_dataset(
        signals=training_signals
    )
    X_testing, y_testing, _ = cnn.scaling_dataset(signals=validation_signals)

    print("training and testing")
    cnn.training_testing(
        X_train=X_training,
        y_train=y_training,
        X_val=X_testing,
        y_val=y_testing,
        classes_weight=classes_weights,
        model=building_model(),
        path="VG/Experiment_2 - Segmentation/CNN/images/small_230",
    )
