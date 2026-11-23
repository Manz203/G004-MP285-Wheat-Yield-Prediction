
import hashlib
from pathlib import Path

import pandas as pd
import mlflow
import mlflow.sklearn

from mlflow import MlflowClient

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.mp285.cleaning import clean_training_data
from src.mp285.preprocessing import (
    build_preprocessor,
    FEATURE_COLUMNS,
)


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
# 4. MLFLOW CONFIGURATION
# --------------------------------------------------

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_registry_uri("sqlite:///mlflow.db")
mlflow.set_experiment("MP285_Wheat_Yield_Prediction")

registered_model_name = "MP285_Wheat_Yield_Prediction_Model"

best_model_uris = {}


# --------------------------------------------------
# 5. MODEL TRAINING AND MLFLOW LOGGING
# --------------------------------------------------

results = []
trained_pipelines = {}

for model_name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", model),
    ])

    with mlflow.start_run(run_name=model_name):

        # Train complete pipeline
        pipeline.fit(X_train, y_train)

        # Generate validation predictions
        predictions = pipeline.predict(X_val)

        # Calculate metrics
        mae = mean_absolute_error(y_val, predictions)

        rmse = mean_squared_error(
            y_val, predictions
        ) ** 0.5

        r2 = r2_score(y_val, predictions)

        # Store results
        results.append({
            "model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        })

        trained_pipelines[model_name] = pipeline

        # Log parameters
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("training_rows", len(X_train))
        mlflow.log_param("validation_rows", len(X_val))
        mlflow.log_param("data_sha256", data_sha256)

        # Log model-specific parameters
        mlflow.log_params(model.get_params())

        # Log metrics
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)

        # Log complete pipeline
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            serialization_format="cloudpickle",
        )

        # Save logged model URI for registration
        best_model_uris[model_name] = (
            mlflow.get_artifact_uri("model")
        )

        print(f"\n{model_name}")
        print("MAE:", mae)
        print("RMSE:", rmse)
        print("R2:", r2)


# --------------------------------------------------
# 6. SELECT BEST MODEL
# --------------------------------------------------

best_result = min(
    results,
    key=lambda result: result["MAE"]
)

best_model_name = best_result["model"]

best_pipeline = trained_pipelines[best_model_name]

print("\nBEST MODEL:", best_model_name)
print("Best MAE:", best_result["MAE"])
print("Best RMSE:", best_result["RMSE"])
print("Best R2:", best_result["R2"])


# --------------------------------------------------
# 7. SAVE BEST MODEL LOCALLY
# --------------------------------------------------

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / "best_model_v1"

mlflow.sklearn.save_model(
    best_pipeline,
    path=str(BEST_MODEL_PATH),
    serialization_format="cloudpickle",
)

print("\nBest model saved to:", BEST_MODEL_PATH)


# --------------------------------------------------
# 8. SAVE MODEL COMPARISON REPORT
# --------------------------------------------------

results_df = pd.DataFrame(results)

REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

results_df.to_csv(
    REPORT_DIR / "model_comparison.csv",
    index=False,
)

print("\nModel comparison saved to reports/model_comparison.csv")


# --------------------------------------------------
# 9. REGISTER BEST MODEL IN MLFLOW
# --------------------------------------------------

model_uri = best_model_uris[best_model_name]

registered_model = mlflow.register_model(
    model_uri=model_uri,
    name=registered_model_name,
)

client = MlflowClient()

client.set_registered_model_alias(
    registered_model_name,
    "champion",
    registered_model.version,
)

print("\nMODEL REGISTRATION SUCCESSFUL")
print("Model:", registered_model_name)
print("Version:", registered_model.version)
print("Alias: champion")


# --------------------------------------------------
# 10. FINAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 50)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 50)

print("Best model:", best_model_name)
print("MAE:", best_result["MAE"])
print("RMSE:", best_result["RMSE"])
print("R2:", best_result["R2"])
print("Registered model:", registered_model_name)
print("Registered version:", registered_model.version)
print("Alias: champion")
