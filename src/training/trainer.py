import os
import tensorflow as tf

from src.data.data_loader import EuroSATDataLoader
from src.helpers.config import get_settings
from src.models.efficientnet import EfficientNetModel

settings = get_settings()


class Trainer:

    def __init__(self):

        self.data_loader = EuroSATDataLoader()

        self.model_builder = EfficientNetModel()

        self.model = self.model_builder.build()

        os.makedirs(
            settings.MODEL_SAVE_DIR,
            exist_ok=True,
        )

    def compile_model(self):

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=settings.LEARNING_RATE,
            ),

            loss="sparse_categorical_crossentropy",

            metrics=[
                "accuracy",
            ],
        )

    def get_callbacks(self):

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(
                settings.MODEL_SAVE_DIR,
                settings.MODEL_FILE_NAME,
            ),

            monitor="val_accuracy",

            save_best_only=True,

            save_weights_only=False,

            mode="max",

            verbose=1,
        )

        early_stopping = tf.keras.callbacks.EarlyStopping(

            monitor="val_loss",

            patience=settings.EARLY_STOPPING_PATIENCE,

            restore_best_weights=True,

            verbose=1,
        )

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(

            monitor="val_loss",

            factor=settings.REDUCE_LR_FACTOR,

            patience=settings.REDUCE_LR_PATIENCE,

            min_lr=settings.MIN_LEARNING_RATE,

            verbose=1,
        )

        return [
            checkpoint,
            early_stopping,
            reduce_lr,
        ]

    def train(self):

        train_dataset, val_dataset, _ = (
            self.data_loader.get_datasets()
        )

        history = self.model.fit(

            train_dataset,

            validation_data=val_dataset,

            epochs=settings.EPOCHS,

            callbacks=self.get_callbacks(),

            verbose=1,
        )

        return history
    
    def fine_tune(self):

        train_dataset, val_dataset, _ = self.data_loader.get_datasets()

        self.model = tf.keras.models.load_model(
            os.path.join(
                settings.MODEL_SAVE_DIR,
                settings.MODEL_FILE_NAME,
            )
        )

        self.model_builder.base_model.trainable = True

        for layer in self.model_builder.base_model.layers[:-settings.UNFREEZE_LAYERS]:
            layer.trainable = False

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=settings.FINE_TUNE_LEARNING_RATE,
            ),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=settings.FINE_TUNE_EPOCHS,
            callbacks=self.get_callbacks(),
            verbose=1,
        )

        return history
    
    def run(self):

        self.compile_model()

        self.train()

        self.fine_tune()