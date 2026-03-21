import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from srs.exception import CustomException
from srs.logger import logging
from srs.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

    train_data_path = os.path.join(base_dir, "artifacts", "train.csv")
    test_data_path = os.path.join(base_dir, "artifacts", "test.csv")
    raw_data_path = os.path.join(base_dir, "artifacts", "data.csv")


class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info("Starting Data Ingestion")

            file_path = os.path.join(
                self.config.base_dir, "srs", "notebook", "DATA", "stud.csv"
            )

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            df = pd.read_csv(file_path)

            os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)

            df.to_csv(self.config.raw_data_path, index=False)

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.config.train_data_path, index=False)
            test_set.to_csv(self.config.test_data_path, index=False)

            logging.info("Data Ingestion Completed")

            return self.config.train_data_path, self.config.test_data_path

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()

    from srs.components.data_transformation import DataTransformation

    transformer = DataTransformation()
    train_arr, test_arr, preprocessor_path = transformer.initiate_data_transformation(
        train_path, test_path
    )

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))