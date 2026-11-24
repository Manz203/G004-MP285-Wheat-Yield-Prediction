
import hashlib
from pathlib import Path

import joblib
import pandas as pd

from mlflow import MlflowClient

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)

from src.mp285.cleaning import clean_training_data
from src.mp285.preprocessing import (
    build_preprocessor,
    FEATURE_COLUMNS,
)
from src.mp285.tracking import (
    setup_tracking,
    run_experiment,
    promote_best,
    load_champion,
)


EXPERIMENT_NAME = "MP285_Wheat_Yield_Prediction"
REGISTERED_MODEL_NAME = "MP285_Wheat_Yield_Prediction_Model"


# --------------------------------------------------
# 1. DATA LOADING AND INTEGRITY CHECK
# --------------------------------------------------

DATA_PATH = Path("data/raw/train.csv")

data_sha256 = hashlib.sha256(
    DATA_PATH.read_bytes()
).hexdigest()

print("Training data SHA-256:", data_sha256)

raw = pd.read_csv(DATA_PATH)

df = clean_training_data(raw)

X = df[FEATURE_COLUMNS]
y = df["yield_qtl_per_acre"]


# --------------------------------------------------
# 2. TRAIN-VALIDATION SPLIT
# --------------------------------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)


# --------------------------------------------------
# 3. MODEL DEFINITIONS
# --------------------------------------------------

models = {
    "linear_regression": LinearRegression(),

    "random_forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    ),

    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    ),
}


# --------------------------------------------------
# 4. MODEL TRAINING AND MLFLOW LOGGING
# --------------------------------------------------

setup_tracking(EXPERIMENT_NAME)

results = []

for model_name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", model),
    ])

    params = {
        "model_name": model_name,
        "test_size": 0.2,
        "random_state": 42,
        "training_rows": len(X_train),
        "validation_rows": len(X_val),
        "data_sha256": data_sha256,
        **model.get_params(),
    }

    run_id, mae = run_experiment(
        model_name, pipeline, params,
        X_train, y_train, X_val, y_val,
    )

    metrics = MlflowClient().get_run(run_id).data.metrics

    results.append({
        "model": model_name,
        "MAE": metrics["mae"],
        "RMSE": metrics["rmse"],
        "R2": metrics["r2"],
    })

    print(f"\n{model_name}")
    print("MAE:", metrics["mae"])
    print("RMSE:", metrics["rmse"])
    print("R2:", metrics["r2"])


# --------------------------------------------------
# 5. SAVE MODEL COMPARISON REPORT
# --------------------------------------------------

REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

pd.DataFrame(results).to_csv(
    REPORT_DIR / "model_comparison.csv",
    index=False,
)

print("\nModel comparison saved to reports/model_comparison.csv")


# --------------------------------------------------
# 6. REGISTER BEST MODEL AND SET THE CHAMPION ALIAS
# --------------------------------------------------

best_run_id, version = promote_best(
    EXPERIMENT_NAME, REGISTERED_MODEL_NAME
)

best_name = min(results, key=lambda r: r["MAE"])["model"]


# --------------------------------------------------
# 7. SAVE THE CHAMPION FOR THE API
# --------------------------------------------------

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(
    load_champion(REGISTERED_MODEL_NAME),
    MODEL_DIR / "model.joblib",
)

print("\nChampion saved to models/model.joblib")


# --------------------------------------------------
# 8. FINAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 50)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 50)

print("Best model:", best_name)
print("Best run id:", best_run_id)
print("Registered model:", REGISTERED_MODEL_NAME)
print("Registered version:", version)
print("Alias: champion")
