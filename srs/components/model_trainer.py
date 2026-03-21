import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

from srs.exception import CustomException
from srs.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    trained_model_file_path = os.path.join(base_dir, "artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False),
                "AdaBoost": AdaBoostRegressor(),
            }

            params = {
                "Random Forest": {"n_estimators": [16, 32]},
                "Decision Tree": {"criterion": ["squared_error"]},
                "Gradient Boosting": {"learning_rate": [0.1], "n_estimators": [16]},
                "Linear Regression": {},
                "XGBRegressor": {"learning_rate": [0.1], "n_estimators": [16]},
                "CatBoost": {"iterations": [30], "depth": [6]},
                "AdaBoost": {"n_estimators": [16]},
            }

            model_report, trained_models = evaluate_models(
                X_train, y_train, X_test, y_test, models, params
            )

            best_model_name = max(model_report, key=model_report.get)
            best_model = trained_models[best_model_name]

            os.makedirs(os.path.dirname(self.config.trained_model_file_path), exist_ok=True)

            save_object(self.config.trained_model_file_path, best_model)

            y_pred = best_model.predict(X_test)
            score = r2_score(y_test, y_pred)

            print(f"Best Model: {best_model_name}, Score: {score}")

            return score

        except Exception as e:
            raise CustomException(e, sys)