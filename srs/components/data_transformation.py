import sys
import os
from dataclasses import dataclass

from srs.exception import CustomException
from srs.logger import logging


@dataclass
class DataTransformationConfig:
    pass


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Data Transformation started")

            # Dummy logic (for now)
            train_arr = train_path
            test_arr = test_path

            logging.info("Data Transformation completed")

            return train_arr, test_arr, None

        except Exception as e:
            raise CustomException(e, sys)