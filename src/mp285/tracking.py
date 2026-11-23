"""MLflow tracking and model registry helpers.

Usage in a training script:

    setup_tracking()
    run_experiment(name, model, params, X_train, y_train, X_val, y_val)
    promote_best()
"""

import mlflow
from mlflow import MlflowClient
from sklearn.metrics import mean_absolute_error

EXPERIMENT_NAME = "wheat_yield"
MODEL_NAME = "wheat_yield_model"
CHAMPION_ALIAS = "champion"
TRACKING_URI = "sqlite:///mlflow.db"


def setup_tracking(experiment_name=EXPERIMENT_NAME, tracking_uri=TRACKING_URI):
    """Point MLflow at the store and select the experiment."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def run_experiment(name, model, params, X_train, y_train, X_val, y_val):
    """Train one model inside an MLflow run.

    The model should be the full pipeline (preprocessor + estimator)
    so that the registered model accepts raw columns.
    Returns (run_id, mae).
    """
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params(params)

        model.fit(X_train, y_train)
        mae = mean_absolute_error(y_val, model.predict(X_val))

        mlflow.log_metric("mae", mae)
        # cloudpickle because the default skops format rejects some
        # types used by the preprocessing pipeline
        mlflow.sklearn.log_model(
            model, name="model", serialization_format="cloudpickle"
        )

        return run.info.run_id, mae


def get_best_run(experiment_name=EXPERIMENT_NAME):
    """Return (run_id, mae) of the run with the lowest MAE."""
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["metrics.mae ASC"],
        max_results=1,
    )

    if runs.empty:
        raise ValueError(f"No finished runs in experiment '{experiment_name}'")

    best = runs.iloc[0]
    return best["run_id"], best["metrics.mae"]


def register_model(run_id, model_name=MODEL_NAME):
    """Register the model saved in a run and return its version."""
    result = mlflow.register_model(f"runs:/{run_id}/model", model_name)
    return int(result.version)


def set_champion(model_name, version, alias=CHAMPION_ALIAS):
    """Point the alias at one version of the registered model."""
    MlflowClient().set_registered_model_alias(model_name, alias, str(version))


def promote_best(experiment_name=EXPERIMENT_NAME, model_name=MODEL_NAME):
    """Register the best run and make it the champion.

    Returns (run_id, version).
    """
    run_id, _ = get_best_run(experiment_name)
    version = register_model(run_id, model_name)
    set_champion(model_name, version)

    return run_id, version


def load_champion(model_name=MODEL_NAME, alias=CHAMPION_ALIAS):
    """Load the model version that currently has the champion alias."""
    return mlflow.sklearn.load_model(f"models:/{model_name}@{alias}")
