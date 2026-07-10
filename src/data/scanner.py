from pathlib import Path

import pandas as pd

from src.constants.eurosat import CLASS_NAMES
from src.helpers.config import get_settings

settings = get_settings()


def scan_dataset() -> pd.DataFrame:
    """
    Scan the EuroSAT dataset and return a dataframe
    containing image paths and labels.
    """

    dataset_dir = Path(settings.DATASET_DIR)
    print(dataset_dir)
    records = []

    for class_name in CLASS_NAMES:

        class_dir = dataset_dir / class_name

        for image_path in class_dir.iterdir():

            if image_path.is_file():

                records.append(
                    {
                        "image_path": str(image_path),
                        "label": class_name,
                    }
                )

    dataframe = pd.DataFrame(records)

    return dataframe