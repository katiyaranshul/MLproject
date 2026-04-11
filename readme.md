# Student Performance Predictor

Predicts **math score** from demographics + reading/writing scores. Pipeline: ingest → preprocess → compare regressors → save best model → **Flask** API/UI.

**Repo:** [github.com/katiyaranshul/student-performance-predictor](https://github.com/katiyaranshul/student-performance-predictor)

## Stack

Python · pandas · scikit-learn · XGBoost · CatBoost · Flask · Jupyter

## Quick start

```bash
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt && pip install -e .
```

**Train** (writes `artifacts/*.csv`, `preprocessor.pkl`, `model.pkl`):

```bash
python srs/components/data_ingestion.py
```

**Run app** → [http://127.0.0.1:5001](http://127.0.0.1:5001)

```bash
python app.py
```

## Layout

| Path | Role |
|------|------|
| `srs/notebook/DATA/stud.csv` | Source data |
| `srs/components/` | Ingestion, transformation, training |
| `srs/pipeline/predict_pipeline.py` | Load artifacts & predict |
| `templates/` | Flask HTML |

**Notebooks:** `EDA STUDENT PERFORMANCE .ipynb`, `MODEL TRAINING.ipynb` (under `srs/notebook/DATA/`).
