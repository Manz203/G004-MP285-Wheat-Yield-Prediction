
import pandas as pd

from src.mp285.cleaning import (
    normalize_seed_variety,
    clean_features,
    clean_training_data,
)


def test_seed_variety_normalization():
    assert normalize_seed_variety(" Hd-2967 ") == "HD-2967"
    assert normalize_seed_variety("DBW.") == "DBW-187"
    assert normalize_seed_variety("Pbw-343") == "PBW-343"


def test_irrigation_sentinel():
    df = pd.DataFrame({
        "seed_variety": ["HD-2967", "DBW-187"],
        "irrigation_count": [-1, 3],
    })

    cleaned = clean_features(df)

    assert pd.isna(cleaned.loc[0, "irrigation_count"])
    assert cleaned.loc[1, "irrigation_count"] == 3
    assert (cleaned["irrigation_count"] == -1).sum() == 0


def test_target_typo_correction():
    df = pd.DataFrame({
        "record_id": ["R00026", "R99999"],
        "seed_variety": ["HD-2967", "DBW-187"],
        "irrigation_count": [2, 3],
        "yield_qtl_per_acre": [250.0, 25.0],
    })

    cleaned = clean_training_data(df)

    assert cleaned.loc[0, "yield_qtl_per_acre"] == 25.0
    assert cleaned.loc[1, "yield_qtl_per_acre"] == 25.0


def test_original_data_unchanged():
    df = pd.DataFrame({
        "seed_variety": [" Hd-2967 "],
        "irrigation_count": [-1],
    })

    original = df.copy(deep=True)
    clean_features(df)

    pd.testing.assert_frame_equal(df, original)



def test_preprocessing_imputation():
    import numpy as np

    from src.mp285.preprocessing import (
        build_preprocessor,
        FEATURE_COLUMNS,
    )

    train = pd.DataFrame({
        "rainfall_mm": [100, 120, 140],
        "fertilizer_kg": [30, 40, 50],
        "sowing_delay_days": [2, 3, 4],
        "irrigation_count": [2, -1, 4],
        "soil_ph": [6.5, 7.0, 7.5],
        "farm_acres": [1, 2, 3],
        "seed_variety": ["HD-2967", "DBW-187", "PBW-343"],
        "tehsil": ["Tehsil North", "Tehsil South", "Tehsil West"],
    })

    validation = train.iloc[[0]].copy()
    validation["irrigation_count"] = -1

    train = clean_features(train)
    validation = clean_features(validation)

    preprocessor = build_preprocessor()

    X_train = preprocessor.fit_transform(train[FEATURE_COLUMNS])
    X_val = preprocessor.transform(validation[FEATURE_COLUMNS])

    imputer = preprocessor.named_transformers_["numeric"].named_steps["imputer"]

    # Median of training values 2 and 4 should be 3.
    assert imputer.statistics_[3] == 3

    # No missing values after preprocessing.
    assert np.isfinite(X_train).all()
    assert np.isfinite(X_val).all()


