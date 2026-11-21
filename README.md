# MP285 — Wheat Yield per Acre (Dehradun)

**SCSE3040 MLOps Mini Project | Group G004 | Batch 2**

## Group details

- **Group ID:** G004
- **Problem ID:** MP285
- **Faculty:** Amit Soni
- **Batch:** 2

### Team members

| Enrolment number | Name |
| --- | --- |
| S24CSEU0222 | Manaswita Shrivastav |
| S24CSEU0133 | Ayush Singh |
| S24CSEU0067 | Lakhan Gupta |

## Problem statement

The client, **Dehradun Kisan Producer Company** (a fictional organisation), needs an estimate of wheat yield for each farm before harvest to plan transportation and storage and identify potentially underperforming fields.

The goal is to predict **`yield_qtl_per_acre`** (quintals per acre), a regression target. The main evaluation metric is **Mean Absolute Error (MAE)**, measured in quintals per acre.

## Project data

- **`train.csv`** — January–June 2026 training records; contains data-quality issues to investigate.
- **`later.csv`** — July–August 2026 records for drift analysis and retraining.
- **`holdout_features.csv`** — September 2026 records without target values, for final predictions.

Raw training datasets are version-controlled using **DVC**, not committed directly to Git.

## MLOps workflow

1. **Data and versioning:** GitHub, DVC, seven data-quality investigations, and `reports/findings.yaml`.
2. **Training and experiments:** Data cleaning, model training, MLflow tracking and model registry.
3. **Serving and automation:** FastAPI prediction service, Docker, pytest, GitHub Actions, Kubernetes manifests, and structured logging.
4. **Monitoring and retraining:** Population Stability Index (PSI), group-wise MAE, v2 retraining and promotion decision, and holdout predictions.

## Stage 1 checklist (18–20 November 2026)

- [x] Create the public GitHub repository.
- [x] Add group and enrolment details to README.
- [x] Initialize DVC and commit dataset tracking metadata.
- [x] Investigate seven possible data problems in `train.csv`.
- [x] Complete the data section of `reports/findings.yaml`.

## Submission

One public GitHub repository is used for the group project. Each member must submit the required form individually by **29 November 2026, 11:59 PM**, and contribute commits using their own GitHub account.

> Source: Assignment `GROUP.md`, `PROBLEM.md`, and `STUDENT_HANDBOOK.md`.
