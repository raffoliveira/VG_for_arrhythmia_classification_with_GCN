import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from collections import defaultdict
from typing import Tuple, Type
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from keras.utils import to_categorical
from tensorflow import keras


class CNN:

    def __init__(self):
        ...

    def segmentation_signals(
        self,
        path: str,
        list_ecgs: list,
        size_beat_before: int,
        size_beat_after: int,
        set_name: str
    ) -> defaultdict:
        """
        Responsible for segmenting ECG signals into beats"

        Args:
            path: path of ECG signals
            list_ecgs: name of each file of ECG signal
            size_beat_before: amount of points before peak of each beat 
            size_beat_after: amount of points after peak of each beat
            set_name: name of set beats

        Returns:
            defaultdict: Dictionary containing all beats divided by class
        """

        dict_signals = defaultdict(list)

        for file in list_ecgs:

            struct = loadmat(os.path.join(path, set_name, file))
            data = struct["individual"][0][0]  # loading info of the signal

            beat_peaks = data["anno_anns"]  # reading R-peak
            beat_types = data["anno_type"]  # reading type of beat

            ecg = data["signal_r"][:, 0]  # reading lead II

            if file == "114.mat":
                ecg = data["signal_r"][:, 1]

            for peak, beat_type in zip(beat_peaks, beat_types):

                beat_sample = []

                if (peak - size_beat_before) < 0 or (peak + size_beat_after) > len(ecg):
                    continue

                if beat_type not in "NLRejAaJSVEFP/fUQ":
                    continue

                beat_sample = ecg[int(peak - size_beat_before): int(peak + size_beat_after)]

                if beat_type in "NLRej":
                    dict_signals["N"].append(beat_sample)
                elif beat_type in "AaJS":
                    dict_signals["S"].append(beat_sample)
                elif beat_type in "VE":
                    dict_signals["V"].append(beat_sample)

        return dict_signals

    def sampling_windows_beats(self, signals: defaultdict) -> defaultdict:
        """
        Sampling of beats from class N

        Args:
            signals: Dictionary of beats

        Returns:
            Same dictionary with sampled class N
        """

        select_beats = []
        for index, beat in enumerate(signals["N"], 1):
            if (index % 10) == 0:
                select_beats.append(beat)

        signals["N"] = select_beats

        return signals

    def scaling_dataset(self, signals: dict) -> Tuple[np.asarray, np.asarray, np.array]:
        """
        Scales the dataset.

        Parameters:
            signals: Dictionary containing ECG signals.

        Returns:
            Scaled dataset (X, y) and class weights.
        """

        ecg_list = []
        ecg_labels = []
        classes = ["N", "S", "V"]

        for class_, beats in signals.items():
            for beat in beats:
                ecg_list.append(np.asarray(beat).reshape(-1, 1))
                ecg_labels.append(classes.index(class_))

        res = compute_class_weight(class_weight="balanced", classes=np.unique(ecg_labels), y=ecg_labels)

        classes_weights = {}
        for i, j in enumerate(res):
            classes_weights.update({i: j})

        X = np.asarray(ecg_list).astype(np.float32).reshape(-1, 280, 1)
        y = np.asarray(ecg_labels).astype(np.float32).reshape(-1, 1)
        y = to_categorical(y)

        return X, y, classes_weights

    def plot_history_metrics(self, history: Type[keras.callbacks.History], cnn_name: str) -> None:
        """
        Plots the history metrics of a trained model.

        Args:
            history: History object returned by model training.
            cnn_name: Name of the CNN model.
        """

        for key, value in history.history.items():
            plt.figure(figsize=(12, 6))
            plt.plot(range(len(value)), value)
            plt.title(str(key))
            plt.savefig(f"./images/{cnn_name}/{str(key)}.png", dpi=600)

    def training_testing(
        self,
        X_train: np.asarray,
        y_train: np.asarray,
        X_val: np.asarray,
        y_val: np.asarray,
        classes_weight: np.array,
        model: Type[keras.Model],
        cnn_name: str
    ) -> None:
        """
        Trains and evaluates the model.

        Args:
            X_train: Training data.
            y_train: Training labels.
            X_val: Validation data.
            y_val: Validation labels.
            classes_weight Weight values for each class.
            model: Keras model to train and evaluate.
            cnn_name: Name of the CNN model.
        """

        epochs = 150
        callbacks = [
            keras.callbacks.ReduceLROnPlateau(
                monitor="categorical_accuracy",
                factor=0.2,
                patience=5,
                min_lr=0.000001
            ),
            keras.callbacks.EarlyStopping(
                monitor="loss",
                patience=10,
                verbose=1
            )
        ]

        optimizer = keras.optimizers.Adam(amsgrad=True, learning_rate=0.001)
        loss = keras.losses.CategoricalCrossentropy()
        metrics = [
            keras.metrics.CategoricalAccuracy(),
            keras.metrics.AUC(),
            keras.metrics.Precision(),
            keras.metrics.Recall()
        ]

        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )

        model_history = model.fit(
            x=X_train,
            y=y_train,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=classes_weight,
            verbose=1
        )

        self.plot_history_metrics(history=model_history, cnn_name=cnn_name)

        loss, accuracy, auc, precision, recall = model.evaluate(
            x=X_val,
            y=y_val,
            verbose=1
        )

        prediction_proba = model.predict(X_val)
        prediction = np.argmax(prediction_proba, axis=1)
        y_true = np.argmax(y_val, axis=1)

        sns.heatmap(
            confusion_matrix(y_true, prediction),
            xticklabels=["N", "S", "V"],
            yticklabels=["N", "S", "V"],
            annot=True,
            fmt=".0f",
            cmap="rocket_r"
        )

        plt.savefig(f"./images/{cnn_name}/confusion_matrix_{cnn_name}.png", dpi=600)

        with open(f"./images/{cnn_name}/report_{cnn_name}.txt", "w") as f:
            f.write(classification_report(y_true, prediction, zero_division=0))

        with open(f"./images/{cnn_name}/{cnn_name}.txt", "w") as f:
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
