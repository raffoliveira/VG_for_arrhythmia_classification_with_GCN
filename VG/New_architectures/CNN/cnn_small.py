import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Type

from json import JSONEncoder
from collections import defaultdict
from scipy.io import loadmat
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils import to_categorical


def segmentation_signals(path: str,
                         list_ecgs: list,
                         size_beat_before: int,
                         size_beat_after: int,
                         set_name: str
                         ) -> defaultdict:

    dict_signals_II = defaultdict(list)  # dict to load beats

    for file in list_ecgs:

        struct = loadmat(os.path.join(path, set_name, file))  # loading the original file
        data = struct["individual"][0][0]  # loading info of the signal

        beat_peaks = data["anno_anns"]  # reading R-peak
        beat_types = data["anno_type"]  # reading type of beat

        ecg_II = data["signal_r"][:, 0]  # reading lead II

        if file == "114.mat":
            ecg_II = data["signal_r"][:, 1]

        for peak, beat_type in zip(beat_peaks, beat_types):

            beat_samples_II = []  # list to save samples of beat II
            # half_beat = int(size_beat/2) #half of size beat

            # if the position is before the begining or
            # if the position is after the ending
            # do nothing
            if (peak - size_beat_before) < 0 or (peak + size_beat_after) > len(ecg_II):
                continue

            # if type of beat is different than this list, do nothing
            if beat_type not in "NLRejAaJSVEFP/fUQ":
                continue

            # taking the samples of beat window
            beat_samples_II = ecg_II[int(peak - size_beat_before): int(peak + size_beat_after)]

            # taking the type of beat and saving in dict
            if beat_type in "NLRej":
                dict_signals_II["N"].append(beat_samples_II)
            elif beat_type in "AaJS":
                dict_signals_II["S"].append(beat_samples_II)
            elif beat_type in "VE":
                dict_signals_II["V"].append(beat_samples_II)

    return dict_signals_II


def sampling_windows_10_beats(signals_II: dict) -> dict:

    select_N_beats_II = []

    for index, beat_II in enumerate(signals_II["N"], 1):
        if (index % 10) == 0:
            select_N_beats_II.append(beat_II)

    signals_II["N"] = select_N_beats_II

    return signals_II


def scaling_dataset(signals_II: dict) -> Tuple[np.asarray, np.asarray, np.array]:

    ecg_list = []
    ecg_labels = []
    classes = ["N", "S", "V"]

    for class_, beats in signals_II.items():
        for beat in beats:
            ecg_list.append(np.asarray(beat).reshape(-1, 1))
            ecg_labels.append(classes.index(class_))

    res = compute_class_weight(class_weight="balanced", classes=np.unique(ecg_labels), y=ecg_labels)
    classes_weights = {}
    [classes_weights.update({i: j}) for i, j in enumerate(res)]

    X = np.asarray(ecg_list).astype(np.float32).reshape(-1, 280, 1)
    y = np.asarray(ecg_labels).astype(np.float32).reshape(-1, 1)
    y = to_categorical(y)

    return X, y, classes_weights


def building_model() -> Type[keras.Model]:

    input_layer = keras.Input(shape=(280, 1))

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


def plot_history_metrics(history: Type[keras.callbacks.History]) -> None:

    for key, value in history.history.items():
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(value)), value)
        plt.title(str(key))
        plt.savefig(f"./images/cnn_small/{str(key)}.png",
                    dpi=600)


class NumpyFloatValuesEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return JSONEncoder.default(self, obj)


def training_testing(X_train: np.asarray, y_train: np.asarray, X_val: np.asarray, y_val: np.asarray,
                     classes_weight: np.array) -> None:

    epochs = 150
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor="categorical_accuracy", factor=0.2, patience=5, min_lr=0.000001),
        keras.callbacks.EarlyStopping(monitor="loss", patience=10, verbose=1)
    ]

    optimizer = keras.optimizers.Adam(amsgrad=True, learning_rate=0.001)
    loss = keras.losses.CategoricalCrossentropy()
    metrics = [keras.metrics.CategoricalAccuracy(),  keras.metrics.AUC(),  keras.metrics.Precision(),
               keras.metrics.Recall()]

    model = building_model()

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    model_history = model.fit(x=X_train, y=y_train, epochs=epochs, callbacks=callbacks,
                              class_weight=classes_weight, verbose=1)

    plot_history_metrics(model_history)

    loss, accuracy, auc, precision, recall = model.evaluate(x=X_val, y=y_val, verbose=1)

    prediction_proba = model.predict(X_val)
    prediction = np.argmax(prediction_proba, axis=1)
    y_true = np.argmax(y_val, axis=1)
    sns.heatmap(confusion_matrix(y_true, prediction),
                xticklabels=["N", "S", "V"],
                yticklabels=["N", "S", "V"],
                annot=True,
                fmt=".0f",
                cmap="rocket_r")
    plt.savefig(f"./images/cnn_small/confusion_matrix_cnn_small.png", dpi=600)

    with open("./images/cnn_small/report_cnn_small.txt", "w") as f:
        f.write(classification_report(y_true, prediction, zero_division=0))

    with open("./images/cnn_small/cnn_small.txt", "w") as f:
        f.write("TRAIN\n")
        f.write(f"Loss: {np.array(model_history.history['loss']).mean()}\n")
        f.write(f"Precision: {np.array(model_history.history['precision']).mean()}\n")
        f.write(f"Recall: {np.array(model_history.history['recall']).mean()}\n")
        f.write(f"AUC: {np.array(model_history.history['auc']).mean()}\n")
        f.write(f"Accuracy: {np.array(model_history.history['categorical_accuracy']).mean()}\n")

        f.write("\nTEST\n")
        f.write(f"Loss: {loss}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"AUC: {auc}\n")
        f.write(f"Accuracy: {accuracy}\n")


if __name__ == "__main__":

    # performing Early_Stopping to find the best number of epoch to training ML models

    PATH = "../../..Data"
    FILES = os.listdir(os.path.join(PATH, "Train"))
    FILES_VAL = ["109.mat", "114.mat", "207.mat", "223.mat"]
    FILES_TRAIN = list(set(FILES)-set(FILES_VAL))

    print("segmentating...")
    train_signals_II = segmentation_signals(PATH, FILES_TRAIN, 100, 180, "Train")
    val_signals_II = segmentation_signals(PATH, FILES_VAL, 100, 180, "Train")

    print("sampling...")
    train_signals_II = sampling_windows_10_beats(train_signals_II)
    val_signals_II = sampling_windows_10_beats(val_signals_II)

    print("scaling")
    X_train, y_train, classes_weights = scaling_dataset(train_signals_II)
    X_test, y_test, _ = scaling_dataset(val_signals_II)

    print("training and testing")
    training_testing(X_train, y_train, X_test, y_test, classes_weights)
