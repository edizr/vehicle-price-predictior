"""Ridge Regression baseline for car price prediction.

This script demonstrates a simple linear baseline (Ridge Regression) on the
CarDekho dataset.
"""

from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 1) Load dataset

# Load dataset (absolute path for robustness)
df = pd.read_csv("c:/Users/atays/Downloads/ML project/ML project/data/cardekho.csv")


# Basic cleaning
df["max_power"] = pd.to_numeric(df["max_power"], errors="coerce")
num_cols = df.select_dtypes(include=["float64", "int64"]).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())


# Feature engineering: create Car_Age before dropping 'year'
df["Car_Age"] = 2025 - df["year"]
df.drop(["year", "name"], axis=1, inplace=True)
df = pd.get_dummies(df, drop_first=True)
X = df.drop('selling_price', axis=1)
y = df['selling_price']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error

ridge_model = Ridge(alpha=1.0)

ridge_model.fit(X_train, y_train)

y_pred_ridge = ridge_model.predict(X_test)

# Train Ridge Regression model
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)

y_pred_ridge = ridge_model.predict(X_test)

# 5) Evaluation
print("Ridge Regression Results")
print("R2:", r2_score(y_test, y_pred_ridge))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_ridge)))

# 5) Evaluation
print("Ridge Regression Results")
print("R2:", r2_score(y_test, y_pred_ridge))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_ridge)))

alphas = [0.01, 0.1, 1, 10, 50, 100]

for a in alphas:
    model = Ridge(alpha=a)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"alpha={a} | R2={r2_score(y_test, y_pred):.4f}")

# Try different alpha values to see effect on performance
alphas = [0.01, 0.1, 1, 10, 50, 100]
for a in alphas:
    model = Ridge(alpha=a)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"alpha={a} | R2={r2_score(y_test, y_pred):.4f}")
