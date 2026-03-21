import sys
from dataclasses import dataclass

from srs.exception import CustomException
from srs.logger import logging


@dataclass
class ModelTrainerConfig:
    pass


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Model training started")

            # Dummy logic
            print("Training model...")

            logging.info("Model training completed")

            return "Model Trained Successfully"

        except Exception as e:
            raise CustomException(e, sys)