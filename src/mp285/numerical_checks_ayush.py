
"""
Member 2 - Ayush
Numerical data quality investigations for MP285 Wheat Yield Prediction.

Checks:
1. unit_mixup
2. sentinel_value
3. stuck_sensor

Run from the project root:
python src/mp285/numerical_checks_ayush.py
"""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "data" / "raw" / "train.csv"

# Conservative investigation thresholds.
UNIT_MIXUP_FACTOR = 5.0
STUCK_SENSOR_MIN_DAYS = 3

# Common special numbers that may represent missing readings.
KNOWN_SENTINELS = {-999, -9999, 9999, 99999}

# Columns that represent counts and cannot normally be negative.
COUNT_COLUMNS = ["irrigation_count"]

# Numerical measurements to investigate for unit differences.
MEASUREMENT_COLUMNS = [
    "rainfall_mm",
    "fertilizer_kg",
    "sowing_delay_days",
    "irrigation_count",
    "soil_ph",
    "farm_acres",
]


def load_data():
    """Load the original training dataset."""
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {TRAIN_PATH}"
        )

    df = pd.read_csv(TRAIN_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df


def check_unit_mixup(df):
    """
    Compare numerical medians across tehsils.

    A factor of 5 or more is reported as a suspicious
    difference requiring manual investigation.
    """
    print("\n" + "=" * 65)
    print("1. UNIT MIXUP INVESTIGATION")
    print("=" * 65)

    found = False

    for col in MEASUREMENT_COLUMNS:
        if col not in df.columns:
            continue

        medians = df.groupby("tehsil")[col].median().dropna()

        print(f"\nColumn: {col}")
        print("Tehsil medians:")
        print(medians.to_string())

        for tehsil, median in medians.items():
            other_medians = medians.drop(tehsil)

            if other_medians.empty:
                continue

            reference = other_medians.median()

            if median == 0 or reference == 0:
                continue

            factor = max(
                abs(median / reference),
                abs(reference / median)
            )

            if factor >= UNIT_MIXUP_FACTOR:
                found = True

                print("\nSUSPICIOUS UNIT DIFFERENCE")
                print("Column:", col)
                print("Affected tehsil:", tehsil)
                print("Tehsil median:", median)
                print("Other-tehsil median:", reference)
                print("Difference factor:", round(factor, 2))

    if not found:
        print(
            "\nNo unit mixup detected using the "
            f"{UNIT_MIXUP_FACTOR}x median comparison rule."
        )

    return found


def check_sentinel_value(df):
    """
    Find suspicious special numerical values.

    Negative irrigation counts are investigated because
    irrigation counts cannot normally be negative.
    """
    print("\n" + "=" * 65)
    print("2. SENTINEL VALUE INVESTIGATION")
    print("=" * 65)

    found = False

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    for col in numeric_columns:
        values = set(KNOWN_SENTINELS)

        if col in COUNT_COLUMNS:
            values.add(-1)

        for value in sorted(values):
            matches = df[df[col] == value]

            if matches.empty:
                continue

            found = True

            print("\nSUSPICIOUS SPECIAL VALUE")
            print("Column:", col)
            print("Value:", value)
            print("Affected records:", len(matches))

            if "tehsil" in matches.columns:
                print("\nAffected tehsils:")
                print(matches["tehsil"].value_counts().to_string())

            if "date" in matches.columns:
                print(
                    "\nDate range:",
                    matches["date"].min(),
                    "to",
                    matches["date"].max()
                )

            print("\nSample evidence:")
            evidence_cols = [
                c for c in [
                    "record_id", "date", "tehsil", col
                ] if c in matches.columns
            ]

            print(
                matches[evidence_cols]
                .head(10)
                .to_string(index=False)
            )

    if not found:
        print("No candidate sentinel values detected.")

    return found


def check_stuck_sensor(df):
    """
    Check for identical daily median readings across
    at least three consecutive calendar dates per tehsil.

    Dates with missing calendar days break a streak.
    """
    print("\n" + "=" * 65)
    print("3. STUCK SENSOR INVESTIGATION")
    print("=" * 65)

    found = False

    valid = df.dropna(subset=["date"]).copy()

    for tehsil, group in valid.groupby("tehsil"):

        print(f"\nTehsil: {tehsil}")

        for col in MEASUREMENT_COLUMNS:
            if col not in group.columns:
                continue

            daily = (
                group.groupby("date")[col]
                .median()
                .dropna()
                .sort_index()
            )

            if daily.empty:
                continue

            run_start = 0

            for i in range(1, len(daily) + 1):

                same_value = (
                    i < len(daily)
                    and daily.iloc[i] == daily.iloc[i - 1]
                )

                consecutive_date = (
                    i < len(daily)
                    and (daily.index[i] - daily.index[i - 1]).days == 1
                )

                if same_value and consecutive_date:
                    continue

                run_length = i - run_start

                if run_length >= STUCK_SENSOR_MIN_DAYS:
                    found = True

                    print("\nPOSSIBLE STUCK SENSOR")
                    print("Column:", col)
                    print("Tehsil:", tehsil)
                    print("Constant daily median:", daily.iloc[run_start])
                    print("Start date:", daily.index[run_start].date())
                    print("End date:", daily.index[i - 1].date())
                    print("Consecutive days:", run_length)

                run_start = i

    if not found:
        print(
            "\nNo stuck sensor detected using the "
            f"{STUCK_SENSOR_MIN_DAYS}-day consecutive-date rule."
        )

    return found


def main():
    """Run all three numerical investigations."""

    print("MP285 - MEMBER 2: AYUSH")
    print("NUMERICAL DATA QUALITY INVESTIGATION")

    df = load_data()

    print("\nDataset:", TRAIN_PATH)
    print("Total records:", len(df))

    unit_issue = check_unit_mixup(df)
    sentinel_issue = check_sentinel_value(df)
    sensor_issue = check_stuck_sensor(df)

    print("\n" + "=" * 65)
    print("FINAL INVESTIGATION SUMMARY")
    print("=" * 65)

    print("unit_mixup:", unit_issue)
    print("sentinel_value:", sentinel_issue)
    print("stuck_sensor:", sensor_issue)

    print("\nInvestigation completed.")


if __name__ == "__main__":
    main()
