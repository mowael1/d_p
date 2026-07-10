from sklearn.model_selection import train_test_split

from src.helpers.config import get_settings


settings = get_settings()


class DatasetSplitter:

    @staticmethod
    def split(dataframe):

        train_df, temp_df = train_test_split(
            dataframe,
            train_size=settings.TRAIN_SPLIT,
            random_state=settings.RANDOM_SEED,
            shuffle=True,
            # stratify=dataframe["label"]
        )

        val_ratio = (
            settings.VAL_SPLIT
            /
            (settings.VAL_SPLIT + settings.TEST_SPLIT)
        )

        val_df, test_df = train_test_split(
            temp_df,
            train_size=val_ratio,
            random_state=settings.RANDOM_SEED,
            shuffle=True,
            # stratify=temp_df["label"]
        )

        return train_df, val_df, test_df