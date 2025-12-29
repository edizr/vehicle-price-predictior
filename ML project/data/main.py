

"""
Car price regression baseline + evaluation.

This script does the following:
- Reads cardekho.csv
- Cleans string columns containing numeric values
- One-hot encodes categorical variables
- Trains a RandomForestRegressor and prints test metrics
- Evaluates generalization with 5-fold cross-validation
- Repeats with log1p-transformed target for comparison
- Plots actual vs predicted prices
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def clean_numeric_column(col: pd.Series) -> pd.Series:
    # Extracts numeric value from strings like '1248 CC'
    if col.dtype == "object":
        extracted = col.astype("string").str.extract(r"(\d+\.?\d*)", expand=False)
        return extracted.astype(float)
    return col

def main():
    # Read dataset
    df = pd.read_csv("c:/Users/atays/Downloads/ML project/ML project/data/cardekho.csv")

    # Drop the 'name' column if present
    if "name" in df.columns:
        df.drop(columns=["name"], inplace=True)

    # Clean numeric string columns
    for col_name in ["mileage(km/ltr/kg)", "engine", "max_power"]:
        if col_name in df.columns:
            df[col_name] = clean_numeric_column(df[col_name])

    # Fill missing values with median and one-hot encode categoricals
    df.fillna(df.median(numeric_only=True), inplace=True)
    df = pd.get_dummies(df, drop_first=True)

    # Features and target
    X = df.drop("selling_price", axis=1)
    y = df["selling_price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train the model
    rf_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)

    # Test set prediction and metrics
    y_pred = rf_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Random Forest Test R2 Score: {r2:.4f}")
    print(f"Random Forest Test MAE: {mae:.2f}")
    print(f"Random Forest Test MSE: {mse:.2f}")

    # 5-fold cross-validation (R2)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf_model, X, y, cv=kf, scoring="r2")
    print(f"5-Fold CV R2 Scores: {cv_scores}")
    print(f"5-Fold CV R2 Mean: {cv_scores.mean():.4f}")

    # Try again with log1p-transformed target
    y_log = np.log1p(y)
    cv_scores_log = cross_val_score(rf_model, X, y_log, cv=kf, scoring="r2")
    print(f"5-Fold CV R2 (log1p target): {cv_scores_log}")
    print(f"5-Fold CV R2 Mean (log1p target): {cv_scores_log.mean():.4f}")

    # Actual vs predicted price plot
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("Actual Selling Price")
    plt.ylabel("Predicted Selling Price")
    plt.title("Actual vs Predicted Selling Price (Test Set)")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

