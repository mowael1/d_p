import cv2
import numpy as np
import tensorflow as tf

from src.constants.eurosat import CLASS_TO_INDEX
from src.data.scanner import scan_dataset
from src.data.split_data import DatasetSplitter
from src.data.transforms import (
    train_transform,
    validation_transform,
)
from src.helpers.config import get_settings

settings = get_settings()


class EuroSATDataLoader:

    def __init__(self):
        self.batch_size = settings.BATCH_SIZE

    def load_dataframes(self):
        """
        Scan the dataset and split it into
        train, validation and test dataframes.
        """

        dataframe = scan_dataset()

        train_df, val_df, test_df = DatasetSplitter.split(dataframe)

        return train_df, val_df, test_df

    def load_image(
        self,
        image_path: str,
        training: bool = True,
    ) -> tf.Tensor:
        """
        Read an image from disk and apply preprocessing.
        """

        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        # OpenCV reads images in BGR format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if training:
            image = train_transform(image)
        else:
            image = validation_transform(image)

        return image

    def create_dataset(
        self,
        dataframe,
        training: bool = True,
    ):
        """
        Convert a dataframe into images and labels.
        """

        images = []
        labels = []

        for _, row in dataframe.iterrows():

            image = self.load_image(
                row["image_path"],
                training=training,
            )

            label = CLASS_TO_INDEX[row["label"]]

            images.append(image)
            labels.append(label)

        images = tf.stack(images)
        labels = tf.convert_to_tensor(
            labels,
            dtype=tf.int32,
        )

        dataset = tf.data.Dataset.from_tensor_slices(
            (images, labels)
        )

        if training:
            dataset = dataset.shuffle(
                buffer_size=len(images),
                seed=settings.RANDOM_SEED,
            )

        dataset = dataset.batch(self.batch_size)

        dataset = dataset.prefetch(
            tf.data.AUTOTUNE,
        )

        return dataset

    def get_datasets(self):
        """
        Return train, validation and test datasets.
        """

        train_df, val_df, test_df = self.load_dataframes()

        train_dataset = self.create_dataset(
            train_df,
            training=True,
        )

        val_dataset = self.create_dataset(
            val_df,
            training=False,
        )

        test_dataset = self.create_dataset(
            test_df,
            training=False,
        )

        return (
            train_dataset,
            val_dataset,
            test_dataset,
        )