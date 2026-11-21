"""Stage 1 checks owned by Lakhan: target typos and spelling variants.

Run from the project root:
python src/mp285/target_checks_lakhan.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "data" / "raw" / "train.csv"
TARGET = "yield_qtl_per_acre"
CATEGORICAL_COLUMNS = ["seed_variety", "tehsil"]


def check_target_typos(df):
    """Find target values that were typed with one extra zero.

    A typo is flagged when the value sits far above the normal spread
    (IQR fence) and dividing it by 10 brings it back inside the range
    of the remaining values.
    """
    print("\n1) TARGET TYPOS")

    values = df[TARGET]
    q1, q3 = values.quantile(0.25), values.quantile(0.75)
    fence = q3 + 1.5 * (q3 - q1)
    print(f"Upper IQR fence: {fence:.2f}")

    outliers = df[values > fence]
    normal = values[values <= fence]
    print(f"Normal range: {normal.min()} to {normal.max()}")

    typos = outliers[
        outliers[TARGET].div(10).between(normal.min(), normal.max())
    ]

    print("\nSuspected typos (value -> corrected):")
    for record_id, value in zip(typos["record_id"], typos[TARGET]):
        print(f"  {record_id}: {value} -> {round(value / 10, 2)}")

    count = len(typos)
    print(f"\nRows typed 10 times too big: {count}")

    if count:
        print(f"findings.yaml -> target_typos: present: true, count: {count}")
    else:
        print("findings.yaml -> target_typos: present: false, count: null")

    return count


def group_key(value):
    """Reduce a label to the part that identifies its category.

    Ignores spaces, case and a truncated tail, so 'HD-.', ' HD-2967'
    and 'Hd-2967' all give the same key.
    """
    label = str(value).strip().upper()
    return label.partition("-")[0].rstrip(".")


def canonical_names(values):
    """Pick the correct spelling for each category.

    Labels are grouped by their key and the spelling used most often
    in that group wins, so 'HD-2967' beats 'Hd-2967' and 'HD-.'.
    This keeps the cleaned labels identical to the ones later.csv and
    holdout_features.csv already use.
    """
    names = {}

    for label in values.value_counts().index:
        key = group_key(label)

        if key not in names:
            names[key] = str(label).strip()

    return names


def clean_category(value, names):
    """Rewrite one label to the correct spelling for its category."""
    return names.get(group_key(value), str(value).strip())


def check_spelling_variants(df):
    """Find the categorical column written in more than one way."""
    print("\n2) SPELLING VARIANTS")

    affected_column = None
    true_levels = None

    for column in CATEGORICAL_COLUMNS:
        raw = df[column].unique()
        names = canonical_names(df[column])
        cleaned = sorted({clean_category(v, names) for v in raw})

        print(f"\nColumn: {column}")
        print(f"Spellings in the file ({len(raw)}): {sorted(raw)}")
        print(f"Real categories ({len(cleaned)}): {cleaned}")

        if len(raw) > len(cleaned):
            affected_column = column
            true_levels = len(cleaned)

    if affected_column:
        print(
            f"\nfindings.yaml -> spelling_variants: present: true, "
            f"column: {affected_column}, true_levels: {true_levels}"
        )
    else:
        print("\nfindings.yaml -> spelling_variants: present: false")

    return affected_column, true_levels


def main():
    df = pd.read_csv(TRAIN_PATH)
    print(f"Training records: {len(df)}")

    check_target_typos(df)
    check_spelling_variants(df)


if __name__ == "__main__":
    main()
