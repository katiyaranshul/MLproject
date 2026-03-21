import os
import sys
import pickle

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

from srs.exception import CustomException


def save_object(file_path, obj):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        trained_models = {}

        for model_name, model in models.items():

            para = param[model_name] if model_name in param else {}

            try:
                gs = GridSearchCV(model, para, cv=3)
                gs.fit(X_train, y_train)

                best_model = gs.best_estimator_

            except Exception:
                # fallback if GridSearch fails
                model.fit(X_train, y_train)
                best_model = model

            y_pred = best_model.predict(X_test)
            score = r2_score(y_test, y_pred)

            report[model_name] = score
            trained_models[model_name] = best_model

        return report, trained_models

    except Exception as e:
        raise CustomException(e, sys)