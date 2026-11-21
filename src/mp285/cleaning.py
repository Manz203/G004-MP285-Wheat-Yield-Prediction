
import numpy as np
import pandas as pd


def normalize_seed_variety(value):
    """Convert spelling variations to standard seed variety names."""
    value = str(value).strip().upper()

    if value.startswith("HD"):
        return "HD-2967"
    elif value.startswith("DBW"):
        return "DBW-187"
    elif value.startswith("PBW"):
        return "PBW-343"

    raise ValueError(f"Unknown seed variety: {value}")


def clean_features(df):
    """Clean seed varieties and irrigation sentinel values."""
    cleaned = df.copy()

    # Standardize seed variety names.
    cleaned["seed_variety"] = cleaned["seed_variety"].apply(
        normalize_seed_variety
    )

    # Convert -1 (unknown irrigation count) to missing.
    cleaned["irrigation_count"] = (
        pd.to_numeric(cleaned["irrigation_count"])
        .replace(-1, np.nan)
    )

    return cleaned


# Records identified during the Stage 1 data-quality investigation.
TARGET_TYPO_IDS = {
    "R00026",
    "R00281",
    "R00433",
    "R00449",
    "R00504",
    "R00624",
    "R00752",
    "R00830",
}


def clean_training_data(df):
    """Clean original training data and correct known target typos."""
    cleaned = clean_features(df)

    # Correct the 8 target values recorded 10 times too large.
    mask = cleaned["record_id"].isin(TARGET_TYPO_IDS)

    cleaned.loc[mask, "yield_qtl_per_acre"] = (
        cleaned.loc[mask, "yield_qtl_per_acre"] / 10
    )

    return cleaned
