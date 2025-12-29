
"""Support Vector Machine regression (SVR) baseline for car price prediction.

This script mirrors the preprocessing steps used in the other models:
1) Load cardekho.csv from the same folder as this script.
2) Clean numeric-looking columns stored as strings.
3) One-hot encode categorical features.
4) Scale features (important for SVM/SVR).
5) Train an SVR model and report metrics.

Note: SVR can be slow on large datasets. If it runs too slowly, try:
- using a smaller subset, or
- switching to LinearSVR.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def clean_numeric_column(col: pd.Series) -> pd.Series:
    # Extract numeric values from string columns like '1248 CC' -> 1248.0
    if col.dtype == "object":
        extracted = col.astype("string").str.extract(r"(\d+\.?\d*)", expand=False)
        return extracted.astype(float)
    return col

def main():
    csv_path = Path(__file__).with_name("cardekho.csv")
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found: {csv_path}. Place 'cardekho.csv' in the same folder as this script."
        )

    df = pd.read_csv(csv_path)

    # 2) Basic preprocessing
    if "name" in df.columns:
        df.drop(columns=["name"], inplace=True)

    # Clean numeric columns that are stored as strings
    for col_name in ["mileage(km/ltr/kg)", "engine", "max_power"]:
        if col_name in df.columns:
            df[col_name] = clean_numeric_column(df[col_name])

    # Fill missing values with median
    df.fillna(df.median(numeric_only=True), inplace=True)

    # 3) One-hot encode categoricals
    df = pd.get_dummies(df, drop_first=True)

    # 4) Train/test split
    X = df.drop("selling_price", axis=1)
    y = df["selling_price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 5) Model
    # Scaling is critical for SVR.
    # Also, car prices are typically right-skewed; training in log space often helps.
    x_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "svr",
                SVR(
                    kernel="rbf",
                    C=100.0,
                    epsilon=0.1,
                    gamma="scale",
                ),
            ),
        ]
    )

    # Use TransformedTargetRegressor to fit in log space and predict in original scale
    model = TransformedTargetRegressor(
        regressor=x_model,
        func=np.log1p,
        inverse_func=np.expm1,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 6) Evaluation
    print("SVR (SVM Regression) RESULTS")
    print("(trained with log1p(target), reported on original price scale)")
    print("R2  :", r2_score(y_test, y_pred))
    print("MAE :", mean_absolute_error(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

    # 7) Plot actual vs predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        linestyle="--",
    )
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("SVR: Actual vs Predicted")
    try:
        plt.show()
    finally:
        plt.close("all")


if __name__ == "__main__":
    main()
