"""
Titanic Survival Prediction — End-to-End Classification Project
Author: Emad Abouelsaad

Demonstrates: data cleaning, preprocessing, feature engineering,
model training, and evaluation using Pandas and Scikit-learn.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
df = pd.read_csv("titanic_raw.csv")
print(f"Raw dataset shape: {df.shape}")
print(f"Missing values before cleaning:\n{df.isnull().sum()[df.isnull().sum() > 0]}\n")

# ---------------------------------------------------------
# 2. DATA CLEANING
# ---------------------------------------------------------
# 'deck' is missing ~77% of the time -> drop it (not usable)
df = df.drop(columns=["deck", "class", "who", "adult_male", "alive", "embark_town"])

# 'age' missing ~20% -> impute with median grouped by passenger class
# (richer/poorer passengers tended to differ in age, so group-median is more accurate than a flat median)
df["age"] = df.groupby("pclass")["age"].transform(lambda x: x.fillna(x.median()))

# 'embarked' missing only 2 rows -> fill with the most common port
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

print(f"Missing values after cleaning:\n{df.isnull().sum().sum()} total missing values remain\n")

# ---------------------------------------------------------
# 3. FEATURE ENGINEERING
# ---------------------------------------------------------
# Family size = siblings/spouses + parents/children + self
df["family_size"] = df["sibsp"] + df["parch"] + 1

# Traveling alone is a strong survival signal
df["is_alone"] = (df["family_size"] == 1).astype(int)

# Fare per person (large families sharing one ticket skews raw fare)
df["fare_per_person"] = df["fare"] / df["family_size"]

# Encode categorical variables
df["sex"] = df["sex"].map({"male": 0, "female": 1})
df = pd.get_dummies(df, columns=["embarked"], drop_first=True)

# Drop columns that don't generalize (raw ticket-level noise) and boolean dtype issues
df["alone"] = df["alone"].astype(int)

features = [
    "pclass", "sex", "age", "sibsp", "parch", "fare",
    "family_size", "is_alone", "fare_per_person", "alone",
    "embarked_Q", "embarked_S",
]
X = df[features]
y = df["survived"]

# ---------------------------------------------------------
# 4. TRAIN / TEST SPLIT + SCALING
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# 5. TRAIN MODELS
# ---------------------------------------------------------
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_scaled, y_train)

rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
rf.fit(X_train, y_train)  # tree models don't need scaling

# ---------------------------------------------------------
# 6. EVALUATE
# ---------------------------------------------------------
def evaluate(name, y_true, y_pred):
    print(f"--- {name} ---")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.3f}")
    print(f"Precision: {precision_score(y_true, y_pred):.3f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.3f}")
    print(f"F1 Score:  {f1_score(y_true, y_pred):.3f}")
    print(classification_report(y_true, y_pred, target_names=["Did not survive", "Survived"]))
    return confusion_matrix(y_true, y_pred)

pred_lr = log_reg.predict(X_test_scaled)
cm_lr = evaluate("Logistic Regression", y_test, pred_lr)

pred_rf = rf.predict(X_test)
cm_rf = evaluate("Random Forest", y_test, pred_rf)

# ---------------------------------------------------------
# 7. FEATURE IMPORTANCE (Random Forest)
# ---------------------------------------------------------
importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
print("Top feature importances (Random Forest):")
print(importances)

# ---------------------------------------------------------
# 8. SAVE VISUALS
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, cm, name in zip(axes, [cm_lr, cm_rf], ["Logistic Regression", "Random Forest"]):
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"{name}\nConfusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["No", "Yes"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["No", "Yes"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=120)

plt.figure(figsize=(8, 5))
importances.plot(kind="barh")
plt.title("Feature Importance — Random Forest")
plt.xlabel("Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=120)

print("\nSaved: confusion_matrices.png, feature_importance.png")
