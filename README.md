# Titanic Survival Prediction — Classification Project

End-to-end machine learning project applying data cleaning, feature engineering,
and classification modeling to predict passenger survival on the Titanic dataset.

## What this project demonstrates
- **Data cleaning**: handled missing values in `age` (group-median imputation by
  passenger class), `embarked` (mode imputation), and dropped `deck` (77% missing,
  not usable).
- **Feature engineering**: created `family_size`, `is_alone`, and `fare_per_person`
  from raw columns; encoded categorical variables (`sex`, `embarked`) for modeling.
- **Model training**: trained and compared two models — Logistic Regression and
  Random Forest Classifier.
- **Evaluation**: accuracy, precision, recall, F1-score, and confusion matrices
  for both models.
- **Interpretability**: extracted feature importance from the Random Forest model
  to identify the strongest predictors of survival.

## Results

| Model               | Accuracy | Precision | Recall | F1-score |
|----------------------|----------|-----------|--------|----------|
| Logistic Regression  | 0.81     | 0.79      | 0.70   | 0.74     |
| Random Forest        | 0.79     | 0.78      | 0.65   | 0.71     |

**Top predictors of survival:** sex, fare / fare per person, age, passenger class —
consistent with the historical "women and children first, and first-class priority"
evacuation pattern.

## Tech stack
Python, Pandas, Scikit-learn, Matplotlib, Seaborn (for dataset loading)

## Files
- `survival_prediction.py` — full pipeline: load → clean → engineer features → train → evaluate
- `confusion_matrices.png` — confusion matrices for both models
- `feature_importance.png` — Random Forest feature importance chart

## How to run
```bash
pip install pandas scikit-learn matplotlib seaborn
python survival_prediction.py
```
