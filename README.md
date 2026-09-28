# Flight Delay Prediction

## Project 21 – Advanced Machine Learning & Deep Learning

**Team ID:** T022  
**Section:** C

## Team Members

| # | Name | Role |
|---|---|---|
| 1 | Samiksha Jangid | Team Lead, Data Pipeline & Leakage Prevention |
| 2 | Rashmi Anand | EDA & Feature Engineering |
| 3 | Ankit Raj Singh | Logistic Regression |
| 4 | Shubhaang Kataruka | XGBoost & Hyperparameter Tuning |
| 5 | Lakshya | Evaluation & Calibration |

## Project Objective

Develop a leakage-safe machine learning system that predicts whether a
scheduled flight will experience substantial delay using information
available before departure.

## Advanced ML Phase

- Data validation and preprocessing
- Temporal and route feature engineering
- Historical delay features
- Logistic Regression baseline
- XGBoost classifier
- Chronological validation
- PR-AUC and ROC-AUC
- Precision and Recall
- Probability calibration
- Operational threshold analysis
- Offline rescheduling analysis

## Future Deep Learning Phase

- Learned airline embeddings
- Origin and destination embeddings
- Temporal modelling
- Comparison with Advanced ML models

## Dataset

The project uses the recommended Kaggle US flight dataset as the
starter dataset. Dataset details and acquisition instructions will be
documented separately.

## Project Structure

```text
configs/       Configuration files
data/          Dataset storage
docs/          Project documentation
models/        Saved model artifacts
notebooks/     Exploration and experiments
reports/       Results and figures
src/           Reusable ML source code
tests/         Automated tests

