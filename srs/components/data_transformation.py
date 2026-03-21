import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from srs.exception import CustomException
from srs.logger import logging
from srs.utils import save_object


@dataclass
class DataTransformationConfig:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    preprocessor_obj_file_path = os.path.join(base_dir, "artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def get_data_transformer_object(self):

        num_cols = ["writing_score", "reading_score"]

        cat_cols = [
            "gender",
            "race_ethnicity",
            "parental_level_of_education",
            "lunch",
            "test_preparation_course",
        ]

        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder()),
            ("scaler", StandardScaler(with_mean=False))
        ])

        return ColumnTransformer([
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols)
        ])

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target = "math_score"

            X_train = train_df.drop(columns=[target])
            y_train = train_df[target]

            X_test = test_df.drop(columns=[target])
            y_test = test_df[target]

            preprocessor = self.get_data_transformer_object()

            X_train = preprocessor.fit_transform(X_train)
            X_test = preprocessor.transform(X_test)

            train_arr = np.c_[X_train, y_train]
            test_arr = np.c_[X_test, y_test]

            os.makedirs(os.path.dirname(self.config.preprocessor_obj_file_path), exist_ok=True)

            save_object(self.config.preprocessor_obj_file_path, preprocessor)

            return train_arr, test_arr, self.config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)