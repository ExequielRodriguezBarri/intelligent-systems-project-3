# ============================================================
# Project 3: Ensemble Learning on Red Wine Quality Dataset
# ============================================================

# If needed, install dependencies (in a notebook):
# !pip install kagglehub scikit-learn matplotlib pandas

import kagglehub
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    BaggingRegressor,
    VotingRegressor
)
from sklearn.metrics import mean_squared_error, r2_score

# ------------------------------------------------------------
# 1. Download dataset with kagglehub
# ------------------------------------------------------------
path = kagglehub.dataset_download("uciml/red-wine-quality-cortez-et-al-2009")
print("Path to dataset files:", path)
print("Files in folder:", os.listdir(path))

# ------------------------------------------------------------
# 2. Load the red wine dataset
# ------------------------------------------------------------
csv_path = os.path.join(path, "winequality-red.csv")
df = pd.read_csv(csv_path)

print("\nData preview:")
print(df.head())
print("\nColumns:", df.columns.tolist())

# ------------------------------------------------------------
# 3. Basic EDA
# ------------------------------------------------------------
print("\nData info:")
print(df.info())

print("\nSummary statistics:")
print(df.describe())

print("\nMissing values per column:")
print(df.isna().sum())

# Plot distribution of target variable (quality)
plt.figure()
df["quality"].hist(bins=10)
plt.title("Distribution of Wine Quality")
plt.xlabel("Quality")
plt.ylabel("Count")
plt.show()

# ------------------------------------------------------------
# 4. Data preprocessing: missing values, outliers, simple feature engineering
# ------------------------------------------------------------

# Handle missing values (drop rows with any missing values)
df_clean = df.dropna().copy()

# Simple outlier handling: clip each numeric feature (except target) to 1st–99th percentiles
numeric_cols = df_clean.columns.drop("quality")

for col in numeric_cols:
    lower = df_clean[col].quantile(0.01)
    upper = df_clean[col].quantile(0.99)
    df_clean[col] = df_clean[col].clip(lower, upper)

# Simple feature engineering example:
# Ratio of free to total sulfur dioxide (add small constant to avoid division by zero)
df_clean["sulfur_dioxide_ratio"] = df_clean["free sulfur dioxide"] / (df_clean["total sulfur dioxide"] + 1e-6)

print("\nAfter cleaning and feature engineering:")
print("Shape:", df_clean.shape)
print("Missing values total:", df_clean.isna().sum().sum())

# ------------------------------------------------------------
# 5. Define features (X) and target (y)
# ------------------------------------------------------------
# Target is the 'quality' column
X = df_clean.drop("quality", axis=1)
y = df_clean["quality"]

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)

# ------------------------------------------------------------
# 6. Train-test split
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain shape:", X_train.shape, "Test shape:", X_test.shape)

# ------------------------------------------------------------
# 7. Define ensemble models
# ------------------------------------------------------------
rf = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

gb = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    random_state=42
)

bag = BaggingRegressor(
    estimator=RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),
    n_estimators=10,
    random_state=42,
    n_jobs=-1
)

voting = VotingRegressor(
    estimators=[
        ("rf", rf),
        ("gb", gb),
        ("bag", bag)
    ]
)

models = {
    "Random Forest": rf,
    "Gradient Boosting": gb,
    "Bagging (RF base)": bag,
    "Voting Regressor": voting
}

# ------------------------------------------------------------
# 8. Train, evaluate, and compare models
# ------------------------------------------------------------
results = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)

    print(f"{name} - MSE: {mse:.4f}, RMSE: {rmse:.4f}, R^2: {r2:.4f}")

    results.append({
        "Model": name,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    })

results_df = pd.DataFrame(results)
print("\nModel comparison:")
print(results_df)

# ------------------------------------------------------------
# 9. Visualize model performance
# ------------------------------------------------------------
plt.figure()
plt.bar(results_df["Model"], results_df["RMSE"])
plt.title("Model Comparison (RMSE)")
plt.ylabel("RMSE (lower is better)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

plt.figure()
plt.bar(results_df["Model"], results_df["R2"])
plt.title("Model Comparison (R²)")
plt.ylabel("R² (higher is better)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 10. Feature importance from Random Forest
# ------------------------------------------------------------
importances = rf.feature_importances_
feature_names = X.columns

feat_imp = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values("importance", ascending=False)

print("\nFeature importances (Random Forest):")
print(feat_imp)

plt.figure()
plt.barh(feat_imp["feature"], feat_imp["importance"])
plt.gca().invert_yaxis()
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()
