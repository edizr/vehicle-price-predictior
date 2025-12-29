"""Ridge Regression alpha sweep (filename kept as-is).

Despite the filename, this script runs Ridge Regression with different alpha
values and prints the R2 score for each.
"""

from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error


# 1) Load dataset
csv_path = Path(__file__).with_name("cardekho.csv")
df = pd.read_csv(csv_path)


# 2) Basic cleaning
# Convert 'max_power' to numeric, coerce errors to NaN
df["max_power"] = pd.to_numeric(df["max_power"], errors="coerce")

# Fill missing values in numeric columns with median
num_cols = df.select_dtypes(include=["int64", "float64"]).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())


# 3) Feature engineering
# Create a new feature for car age
df["Car_Age"] = 2025 - df["year"]
# Drop 'year' and 'name' columns (not needed for regression)
df.drop(["year", "name"], axis=1, inplace=True)

# One-hot encode categorical variables
df = pd.get_dummies(df, drop_first=True)


# Define features and target
X = df.drop('selling_price', axis=1)
y = df['selling_price']


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Decision Tree Regression
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
y_pred = dt_model.predict(X_test)


# 5) Evaluation
print("RIDGE REGRESSION RESULTS")
print("R2:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print(f"Model Performance: {r2_score(y_test, y_pred)*100:.2f}%")

