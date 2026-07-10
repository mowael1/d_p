import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.helpers.config import get_settings
from src.constants.eurosat import NUM_CLASSES
settings = get_settings()


class EfficientNetModel:

    def __init__(self):

        self.image_size = settings.IMAGE_SIZE
        self.num_classes = NUM_CLASSES

        self.base_model = None

    def build(self):

        self.base_model = tf.keras.applications.EfficientNetB3(
            include_top=False,
            weights="imagenet",
            input_shape=(
                self.image_size,
                self.image_size,
                3,
            ),
        )

        # Phase 1
        self.base_model.trainable = False

        inputs = keras.Input(
            shape=(
                self.image_size,
                self.image_size,
                3,
            )
        )

        x = layers.Rescaling(255.0)(inputs)

        x = self.base_model(
            x,
            training=False,
        )

        x = layers.GlobalAveragePooling2D()(x)

        x = layers.BatchNormalization()(x)

        x = layers.Dense(
            512,
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(1e-4),
        )(x)

        x = layers.Dropout(0.5)(x)

        x = layers.Dense(
            256,
            activation="relu",
        )(x)

        x = layers.Dropout(0.3)(x)

        outputs = layers.Dense(
            self.num_classes,
            activation="softmax",
        )(x)

        model = keras.Model(
            inputs=inputs,
            outputs=outputs,
            name="EfficientNetB3_TL",
        )

        return model