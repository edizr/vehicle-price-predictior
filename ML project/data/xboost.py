"""XGBoost regression baseline for car price prediction.

This script follows the typical ML workflow:
1) Load dataset (cardekho.csv) from the same folder.
2) Clean columns that store numbers inside strings (units included).
3) One-hot encode categorical features.
4) Train XGBRegressor and report metrics + feature importances.

Note: requires the external package `xgboost`.
"""

from pathlib import Path


from xgboost import XGBRegressor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 1) Load dataset
csv_path = Path(__file__).with_name("cardekho.csv")
df = pd.read_csv(csv_path)
print(df.columns)

# 2) Basic preprocessing
if "name" in df.columns:
    df.drop(columns=["name"], inplace=True)


def clean_numeric_column(col: pd.Series) -> pd.Series:
    # Extract numeric values from strings (e.g., '1248 CC' -> 1248.0)
    if col.dtype == "object":
        extracted = col.astype("string").str.extract(r"(\d+\.?\d*)", expand=False)
        return extracted.astype(float)
    return col


for col_name in ["mileage(km/ltr/kg)", "engine", "max_power"]:
    if col_name in df.columns:
        df[col_name] = clean_numeric_column(df[col_name])
# Fill missing values with median and one-hot encode categoricals
df.fillna(df.median(numeric_only=True), inplace=True)
df = pd.get_dummies(df, drop_first=True)


# Prepare features and target
X = df.drop("selling_price", axis=1)
y = df["selling_price"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize and train the XGBoost Regressor
xgb_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# 4) Evaluation
print("XGBoost R2  :", r2_score(y_test, y_pred_xgb))
print("XGBoost MAE :", mean_absolute_error(y_test, y_pred_xgb))
print("XGBoost RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_xgb)))

# Show top 10 most important features
xgb_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": xgb_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print(xgb_importance.head(10))

