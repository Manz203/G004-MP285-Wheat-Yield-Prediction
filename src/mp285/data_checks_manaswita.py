"""Stage 1 checks owned by Manaswita: leakage and duplicate rows.

Run from anywhere using: python src/mp285/data_checks_manaswita.py
Source of truth: the original, unmodified MP285 train.csv.
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "data" / "raw" / "train.csv"
HOLDOUT_PATH = ROOT / "data" / "raw" / "holdout_features.csv"
TARGET = "yield_qtl_per_acre"


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    holdout = pd.read_csv(HOLDOUT_PATH)

    print(f"Training records: {len(train)}")
    print(f"Holdout records: {len(holdout)}")

    # A leakage candidate might reveal information that is unavailable
    # when an actual prediction is requested. The label itself is expected
    # to be absent from holdout_features.csv and is NOT a leaking feature.
    missing_from_holdout = sorted(set(train.columns) - set(holdout.columns))
    candidate_columns = [c for c in missing_from_holdout if c != TARGET]
    print("\n1) LEAKING COLUMN CHECK")
    print("Columns missing from holdout:", missing_from_holdout)
    print("Non-target columns missing from holdout:", candidate_columns)

    # Extra diagnostic: near-perfect numerical agreement with the target
    # may indicate a proxy for the answer. It isn't proof by itself.
    numeric_columns = [
        c for c in train.select_dtypes(include="number").columns if c != TARGET
    ]
    correlations = train[numeric_columns].corrwith(train[TARGET]).abs()
    suspicious_correlations = correlations[correlations >= 0.995]
    print("Numerical features with |correlation to target| >= 0.995:",
          suspicious_correlations.index.tolist())
    print("Leakage candidates require manual review; do not use the target as an input.")

    print("\n2) DUPLICATE ROWS CHECK")
    duplicate_count = int(train.duplicated(keep="first").sum())
    print("Exact extra copies:", duplicate_count)
    print("Repeated record_id values:",
          int(train.duplicated(subset=["record_id"], keep="first").sum()))

    # The handbook says when present is false, other fields remain null.
    # The two recommended values below apply if there are no candidates.
    if not candidate_columns and suspicious_correlations.empty:
        print("\nfindings.yaml -> leaking_column: present: false, column: null")
    else:
        print("\nInvestigate potential leakage candidates before filling findings.yaml")

    if duplicate_count == 0:
        print("findings.yaml -> duplicate_rows: present: false, count: null")
    else:
        print(f"findings.yaml -> duplicate_rows: present: true, count: {duplicate_count}")


if __name__ == "__main__":
    main()
