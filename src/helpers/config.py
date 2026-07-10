from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    # =========================
    # Dataset
    # =========================
    DATASET_DIR: str

    # =========================
    # Model
    # =========================
    MODEL_NAME: str
    MODEL_SAVE_DIR: str
    MODEL_FILE_NAME: str

    # =========================
    # Training
    # =========================
    IMAGE_SIZE: int = Field(gt=0)
    BATCH_SIZE: int = Field(gt=0)
    EPOCHS: int = Field(gt=0)
    LEARNING_RATE: float = Field(gt=0)
    RANDOM_SEED: int
    EARLY_STOPPING_PATIENCE: int

    # =========================
    # Fine Tuning
    # =========================
    FINE_TUNE_EPOCHS: int = Field(gt=0)
    FINE_TUNE_LEARNING_RATE: float = Field(gt=0)
    UNFREEZE_LAYERS: int = Field(gt=0)

    # =========================
    # Dataset Split
    # =========================
    TRAIN_SPLIT: float
    VAL_SPLIT: float
    TEST_SPLIT: float

    # =========================
    # Callbacks
    # =========================
    REDUCE_LR_PATIENCE: int = Field(gt=0)
    REDUCE_LR_FACTOR: float = Field(gt=0)
    MIN_LEARNING_RATE: float = Field(gt=0)

    # =========================
    # API
    # =========================
    API_HOST: str
    API_PORT: int

    model_config = SettingsConfigDict(
        env_file="src/.env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()