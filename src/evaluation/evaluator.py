import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

from src.constants.eurosat import INDEX_TO_CLASS
from src.data.data_loader import EuroSATDataLoader
from src.helpers.config import get_settings

settings = get_settings()


class Evaluator:

    def __init__(self):

        self.data_loader = EuroSATDataLoader()

        self.model = tf.keras.models.load_model(
            os.path.join(
                settings.MODEL_SAVE_DIR,
                settings.MODEL_FILE_NAME,
            )
        )

    def evaluate(self):

        _, _, test_dataset = self.data_loader.get_datasets()

        y_true = []
        y_pred = []

        for images, labels in test_dataset:

            predictions = self.model.predict(images, verbose=0)

            predicted_classes = np.argmax(predictions, axis=1)

            y_true.extend(labels.numpy())

            y_pred.extend(predicted_classes)

        accuracy = accuracy_score(y_true, y_pred)

        precision = precision_score(
            y_true,
            y_pred,
            average="weighted",
        )

        recall = recall_score(
            y_true,
            y_pred,
            average="weighted",
        )

        f1 = f1_score(
            y_true,
            y_pred,
            average="weighted",
        )

        report = classification_report(
            y_true,
            y_pred,
            target_names=INDEX_TO_CLASS.values(),
        )

        matrix = confusion_matrix(
            y_true,
            y_pred,
        )

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")

        print("\nClassification Report\n")
        print(report)

        print("\nConfusion Matrix\n")
        print(matrix)