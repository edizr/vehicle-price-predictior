
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
# Use a non-interactive backend for deterministic, headless figure export (avoids Tkinter issues)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.compose import TransformedTargetRegressor

# Try to import XGBoost if available
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False



# Utility class for managing project paths
@dataclass(frozen=True)
class Paths:
    repo_root: Path

    @property
    def data_dir(self) -> Path:
        return self.repo_root / "ML project" / "data"

    @property
    def csv_path(self) -> Path:
        return self.data_dir / "cardekho.csv"

    @property
    def report_dir(self) -> Path:
        return self.repo_root / "reports"

    @property
    def fig_dir(self) -> Path:
        return self.report_dir / "figures"



# Extract numeric values from strings like '1248 CC' -> 1248.0
def clean_numeric_column(col: pd.Series) -> pd.Series:
    if col.dtype == "object":
        extracted = col.astype("string").str.extract(r"(\d+\.?\d*)", expand=False)
        return extracted.astype(float)
    return col



# Load the raw dataset from CSV
def load_raw_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df



# Preprocess the raw dataframe for modeling
# Returns (X, y, df_processed_before_one_hot) for modeling and plotting
def preprocess_for_model(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    df = df_raw.copy()
    if "name" in df.columns:
        df.drop(columns=["name"], inplace=True)
    for col_name in ["mileage(km/ltr/kg)", "engine", "max_power"]:
        if col_name in df.columns:
            df[col_name] = clean_numeric_column(df[col_name])
    # Save a copy before one-hot encoding for category/subtype plots
    df_pre_onehot = df.copy()
    # Median-impute numeric columns
    df.fillna(df.median(numeric_only=True), inplace=True)
    # One-hot encode categorical variables
    df = pd.get_dummies(df, drop_first=True)
    X = df.drop("selling_price", axis=1)
    y = df["selling_price"]
    return X, y, df_pre_onehot



# Save the current matplotlib figure to the given path
def save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=450, bbox_inches="tight")
    plt.close("all")



# Standard slide figure size for presentation (16:9)
SLIDE_FIGSIZE = (16, 9)



# Plot the distribution of the target variable (selling_price)
def plot_target_distribution(y: pd.Series, out_path: Path) -> None:
    plt.figure(figsize=SLIDE_FIGSIZE)
    sns.histplot(y, bins=40, kde=True)
    plt.title("Target Distribution: selling_price")
    plt.xlabel("selling_price")
    plt.ylabel("count")
    save_fig(out_path)



# Plot the log-transformed distribution of the target variable
def plot_target_log_distribution(y: pd.Series, out_path: Path) -> None:
    plt.figure(figsize=SLIDE_FIGSIZE)
    sns.histplot(np.log1p(y), bins=40, kde=True)
    plt.title("Target Distribution: log1p(selling_price)")
    plt.xlabel("log1p(selling_price)")
    plt.ylabel("count")
    save_fig(out_path)



# Plot the percentage of missing values per feature
def plot_missingness(df: pd.DataFrame, out_path: Path) -> None:
    missing_pct = (df.isna().mean() * 100.0).sort_values(ascending=False)
    shown = missing_pct[missing_pct > 0]
    if shown.empty:
        shown = missing_pct.head(min(10, len(missing_pct)))
    if len(shown) > 18:
        shown = shown.head(18)
    plot_df = shown.reset_index()
    plot_df.columns = ["feature", "missing_pct"]
    plt.figure(figsize=SLIDE_FIGSIZE)
    ax = sns.barplot(data=plot_df, y="feature", x="missing_pct", orient="h")
    ax.set_title("Missing Values by Feature (%)")
    ax.set_xlabel("% missing")
    ax.set_ylabel("feature")
    ax.set_xlim(0, max(1.0, float(plot_df["missing_pct"].max()) * 1.15))
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", padding=6)
    save_fig(out_path)



# Plot a correlation heatmap for all numeric features
def plot_numeric_correlation(df: pd.DataFrame, out_path: Path) -> None:
    num = df.select_dtypes(include=["int64", "float64"]).copy()
    if "selling_price" not in num.columns:
        return
    corr = num.corr(numeric_only=True)
    plt.figure(figsize=SLIDE_FIGSIZE)
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        square=False,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Correlation Heatmap (Numeric Features)")
    save_fig(out_path)



# Plot the distribution of categorical subtype features
def plot_category_subtypes(df_pre_onehot: pd.DataFrame, out_path: Path) -> None:
    cols = [c for c in ["fuel", "transmission", "seller_type", "owner"] if c in df_pre_onehot.columns]
    if not cols:
        return
    n = len(cols)
    plt.figure(figsize=(12.8, max(7.2, 3.2 * n)))
    for i, col in enumerate(cols, start=1):
        plt.subplot(n, 1, i)
        order = df_pre_onehot[col].value_counts().index
        sns.countplot(data=df_pre_onehot, y=col, order=order)
        plt.title(f"Category Distribution (Subtype): {col}")
        plt.xlabel("count")
        plt.ylabel(col)
    save_fig(out_path)



# Train a Random Forest regressor with fixed hyperparameters
def train_rf(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_xgb(X_train: pd.DataFrame, y_train: pd.Series):
    if not XGBOOST_AVAILABLE:
        return None
    model = XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_svr_log_target(X_train: pd.DataFrame, y_train: pd.Series):
    x_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svr", SVR(kernel="rbf", C=100.0, epsilon=0.1, gamma="scale")),
        ]
    )
    model = TransformedTargetRegressor(regressor=x_model, func=np.log1p, inverse_func=np.expm1)
    model.fit(X_train, y_train)
    return model


def plot_actual_vs_pred(y_true: pd.Series, y_pred: np.ndarray, title: str, out_path: Path) -> None:
    plt.figure(figsize=SLIDE_FIGSIZE)
    plt.scatter(y_true, y_pred, alpha=0.55, s=22)
    lo, hi = float(np.min(y_true)), float(np.max(y_true))
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=2)
    plt.title(title)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    save_fig(out_path)


def plot_residuals(y_true: pd.Series, y_pred: np.ndarray, title: str, out_path: Path) -> None:
    residuals = y_true.to_numpy() - y_pred
    plt.figure(figsize=SLIDE_FIGSIZE)
    sns.histplot(residuals, bins=50, kde=True)
    plt.title(title)
    plt.xlabel("Residual (Actual - Predicted)")
    plt.ylabel("count")
    save_fig(out_path)


def plot_top_feature_importance(model, feature_names: list[str], out_path: Path, title: str) -> None:
    if not hasattr(model, "feature_importances_"):
        return
    imp = np.asarray(model.feature_importances_)
    idx = np.argsort(imp)[::-1][:20]
    top = pd.DataFrame({"feature": np.array(feature_names)[idx], "importance": imp[idx]})

    plt.figure(figsize=SLIDE_FIGSIZE)
    sns.barplot(data=top, x="importance", y="feature")
    plt.title(title)
    plt.xlabel("importance")
    plt.ylabel("feature")
    save_fig(out_path)


def plot_cv_comparison(X: pd.DataFrame, y: pd.Series, out_path: Path) -> None:
    # Keep CV lightweight so figure generation stays fast.
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )

    svr = train_svr_log_target

    scores = []

    rf_scores = cross_val_score(rf, X, y, cv=cv, scoring="r2", n_jobs=-1)
    scores.extend([("RandomForest", s) for s in rf_scores])

    # SVR is slower; use fewer folds to keep runtime acceptable.
    cv3 = KFold(n_splits=3, shuffle=True, random_state=42)
    svr_model = TransformedTargetRegressor(
        regressor=Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("svr", SVR(kernel="rbf", C=100.0, epsilon=0.1, gamma="scale")),
            ]
        ),
        func=np.log1p,
        inverse_func=np.expm1,
    )
    svr_scores = cross_val_score(svr_model, X, y, cv=cv3, scoring="r2", n_jobs=-1)
    scores.extend([("SVR (3-fold)", s) for s in svr_scores])

    if XGBOOST_AVAILABLE:
        xgb = XGBRegressor(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )
        xgb_scores = cross_val_score(xgb, X, y, cv=cv, scoring="r2", n_jobs=-1)
        scores.extend([("XGBoost", s) for s in xgb_scores])

    df_scores = pd.DataFrame(scores, columns=["model", "r2"])
    plt.figure(figsize=SLIDE_FIGSIZE)
    sns.boxplot(data=df_scores, x="model", y="r2")
    sns.stripplot(data=df_scores, x="model", y="r2", color="black", alpha=0.5)
    plt.title("Cross-Validation R2 Comparison")
    plt.xlabel("model")
    plt.ylabel("R2")
    plt.xticks(rotation=10)
    save_fig(out_path)


def plot_roc_auc_example(X: pd.DataFrame, y: pd.Series, out_roc: Path, out_balance: Path) -> None:
    # ROC-AUC is a classification metric. We create a simple binary label:
    # "expensive" if price >= median price.
    threshold = float(np.median(y))
    y_bin = (y >= threshold).astype(int)

    # Keep this demo lightweight; it's not the project's main evaluation.
    if len(X) > 5000:
        X_demo, _, y_demo, _ = train_test_split(
            X,
            y_bin,
            train_size=5000,
            random_state=42,
            stratify=y_bin,
        )
    else:
        X_demo, y_demo = X, y_bin

    # Class balance plot
    plt.figure(figsize=SLIDE_FIGSIZE)
    sns.countplot(x=y_bin)
    plt.title("Class Balance (Binary label for ROC-AUC demo)")
    plt.xlabel("label (0=below median, 1=above/equal median)")
    plt.ylabel("count")
    save_fig(out_balance)

    X_train, X_test, y_train, y_test = train_test_split(
        X_demo, y_demo, test_size=0.2, random_state=42, stratify=y_demo
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
        min_samples_leaf=5,
    )
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, proba)
    fpr, tpr, _ = roc_curve(y_test, proba)

    plt.figure(figsize=SLIDE_FIGSIZE)
    plt.plot(fpr, tpr, label=f"ROC curve (AUC={auc:.3f})", linewidth=2.5)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=2)
    plt.title("ROC Curve (Demo: converting regression to binary task)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    save_fig(out_roc)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    paths = Paths(repo_root=repo_root)

    if not paths.csv_path.exists():
        raise FileNotFoundError(
            f"Could not find dataset at {paths.csv_path}. Expected it under ML project/data/cardekho.csv"
        )

    paths.fig_dir.mkdir(parents=True, exist_ok=True)

    df_raw = load_raw_dataset(paths.csv_path)
    X, y, df_pre_onehot = preprocess_for_model(df_raw)

    # Global styling (presentation / distant readability)
    sns.set_theme(style="whitegrid")
    sns.set_context("talk", font_scale=1.15)
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.titlesize": 24,
            "axes.labelsize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "lines.linewidth": 2.2,
            "axes.linewidth": 1.2,
        }
    )

    # Basic dataset plots
    plot_target_distribution(y, paths.fig_dir / "01_target_distribution.png")
    plot_target_log_distribution(y, paths.fig_dir / "02_target_log_distribution.png")
    plot_missingness(df_pre_onehot, paths.fig_dir / "03_missingness_heatmap.png")
    plot_numeric_correlation(df_raw, paths.fig_dir / "04_numeric_correlation.png")
    plot_category_subtypes(df_pre_onehot, paths.fig_dir / "05_category_subtypes.png")

    # Train/test split for model plots
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf = train_rf(X_train, y_train)
    rf_pred = rf.predict(X_test)

    plot_actual_vs_pred(
        y_test,
        rf_pred,
        title="RandomForest: Actual vs Predicted",
        out_path=paths.fig_dir / "06_rf_actual_vs_pred.png",
    )
    plot_residuals(
        y_test,
        rf_pred,
        title="RandomForest Residual Distribution",
        out_path=paths.fig_dir / "07_rf_residuals.png",
    )
    plot_top_feature_importance(
        rf,
        list(X.columns),
        out_path=paths.fig_dir / "08_rf_feature_importance_top20.png",
        title="RandomForest Feature Importance (Top 20)",
    )

    if XGBOOST_AVAILABLE:
        xgb = train_xgb(X_train, y_train)
        if xgb is not None:
            xgb_pred = xgb.predict(X_test)
            plot_actual_vs_pred(
                y_test,
                xgb_pred,
                title="XGBoost: Actual vs Predicted",
                out_path=paths.fig_dir / "09_xgb_actual_vs_pred.png",
            )
            plot_top_feature_importance(
                xgb,
                list(X.columns),
                out_path=paths.fig_dir / "10_xgb_feature_importance_top20.png",
                title="XGBoost Feature Importance (Top 20)",
            )

    svr = train_svr_log_target(X_train, y_train)
    svr_pred = svr.predict(X_test)
    plot_actual_vs_pred(
        y_test,
        svr_pred,
        title="SVR (log-target): Actual vs Predicted",
        out_path=paths.fig_dir / "11_svr_actual_vs_pred.png",
    )

    # CV comparison plot
    plot_cv_comparison(X, y, paths.fig_dir / "12_cv_r2_comparison.png")

    # ROC-AUC demo + imbalance discussion plots
    plot_roc_auc_example(
        X,
        y,
        out_roc=paths.fig_dir / "13_roc_auc_demo.png",
        out_balance=paths.fig_dir / "14_binary_class_balance.png",
    )

    # Save a small metrics summary table for the presentation.
    def metrics(y_true: pd.Series, y_hat: np.ndarray) -> dict[str, float]:
        return {
            "r2": float(r2_score(y_true, y_hat)),
            "mae": float(mean_absolute_error(y_true, y_hat)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_hat))),
        }

    rows = []
    rows.append({"model": "RandomForest", **metrics(y_test, rf_pred)})
    if XGBOOST_AVAILABLE and "xgb_pred" in locals():
        rows.append({"model": "XGBoost", **metrics(y_test, xgb_pred)})
    rows.append({"model": "SVR (log-target)", **metrics(y_test, svr_pred)})

    summary = pd.DataFrame(rows)
    summary.to_csv(paths.report_dir / "metrics_summary.csv", index=False)


if __name__ == "__main__":
    main()
