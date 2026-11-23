
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.mp285.preprocessing import build_preprocessor
from src.mp285.tracking import (
    get_best_run,
    load_champion,
    promote_best,
    register_model,
    run_experiment,
    set_champion,
    setup_tracking,
)

EXPERIMENT = "test_experiment"
MODEL = "test_model"


@pytest.fixture(autouse=True)
def tracking(tmp_path, monkeypatch):
    """Use a throwaway MLflow store so tests never touch the real one."""
    monkeypatch.chdir(tmp_path)
    setup_tracking(EXPERIMENT, f"sqlite:///{tmp_path}/mlflow.db")


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(60, 2)), columns=["a", "b"])
    y = 3 * X["a"] - X["b"] + rng.normal(scale=0.1, size=60)
    return X[:40], y[:40], X[40:], y[40:]


def make_model(alpha=1.0):
    return Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=alpha))])


def test_run_experiment_returns_run_id_and_mae(data):
    X_train, y_train, X_val, y_val = data

    run_id, mae = run_experiment(
        "ridge", make_model(), {"alpha": 1.0},
        X_train, y_train, X_val, y_val,
    )

    assert run_id
    assert 0 <= mae < 1


def test_get_best_run_picks_lowest_mae(data):
    X_train, y_train, X_val, y_val = data

    results = []

    # a huge alpha shrinks the weights, so it should score much worse
    for alpha in [1000, 0.1, 10]:
        results.append(run_experiment(
            f"ridge_{alpha}", make_model(alpha), {"alpha": alpha},
            X_train, y_train, X_val, y_val,
        ))

    expected_id, expected_mae = min(results, key=lambda r: r[1])
    best_id, best_mae = get_best_run(EXPERIMENT)

    assert best_id == expected_id
    assert best_mae == pytest.approx(expected_mae)


def test_get_best_run_without_runs_raises():
    with pytest.raises(ValueError):
        get_best_run(EXPERIMENT)


def test_register_model_increments_version(data):
    X_train, y_train, X_val, y_val = data
    run_id, _ = run_experiment(
        "ridge", make_model(), {"alpha": 1.0},
        X_train, y_train, X_val, y_val,
    )

    assert register_model(run_id, MODEL) == 1
    assert register_model(run_id, MODEL) == 2


def test_set_champion_moves_alias(data):
    X_train, y_train, X_val, y_val = data
    run_id, _ = run_experiment(
        "ridge", make_model(), {"alpha": 1.0},
        X_train, y_train, X_val, y_val,
    )
    register_model(run_id, MODEL)
    register_model(run_id, MODEL)

    set_champion(MODEL, 1)
    set_champion(MODEL, 2)

    champion = MlflowClient().get_model_version_by_alias(MODEL, "champion")
    assert int(champion.version) == 2


def test_real_preprocessor_pipeline_can_be_logged_and_loaded():
    rng = np.random.default_rng(1)
    n = 60
    X = pd.DataFrame({
        "rainfall_mm": rng.integers(200, 1200, n),
        "fertilizer_kg": rng.uniform(20, 150, n),
        "sowing_delay_days": rng.integers(0, 40, n),
        "irrigation_count": rng.integers(0, 8, n).astype(float),
        "soil_ph": rng.uniform(5.5, 8.5, n),
        "farm_acres": rng.uniform(0.5, 12, n),
        "seed_variety": rng.choice(["HD-2967", "DBW-187", "PBW-343"], n),
        "tehsil": rng.choice(["Tehsil North", "Tehsil South"], n),
    })
    X.loc[0, "irrigation_count"] = np.nan
    y = X["fertilizer_kg"] / 10 + rng.normal(size=n)

    model = Pipeline([
        ("prep", build_preprocessor()),
        ("model", Ridge()),
    ])

    run_experiment("real_prep", model, {}, X[:40], y[:40], X[40:], y[40:])
    promote_best(EXPERIMENT, MODEL)

    loaded = load_champion(MODEL)
    assert len(loaded.predict(X[40:])) == 20


def test_promote_best_and_load_champion(data):
    X_train, y_train, X_val, y_val = data

    for alpha in [0.1, 1000]:
        run_experiment(
            f"ridge_{alpha}", make_model(alpha), {"alpha": alpha},
            X_train, y_train, X_val, y_val,
        )

    best_id, _ = get_best_run(EXPERIMENT)
    run_id, version = promote_best(EXPERIMENT, MODEL)

    assert run_id == best_id
    assert version == 1

    model = load_champion(MODEL)
    assert len(model.predict(X_val)) == len(X_val)
