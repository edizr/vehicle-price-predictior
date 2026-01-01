# Vehicle Price Predictor

A machine learning project to predict the selling price of used cars using regression models (Ridge Regression, Decision Tree, XGBoost, SVM) on the Cardekho dataset. Includes data preprocessing, model evaluation, and result visualization.

## Project Structure

- `ML project/data/cardekho.csv` — Main dataset
- `ML project/data/main.py` — Main script to run experiments
- `ML project/data/decisiontree.py` — Decision Tree model
- `ML project/data/ridgeregression.py` — Ridge Regression model
- `ML project/data/svm.py` — Support Vector Machine model
- `ML project/data/xboost.py` — XGBoost model
- `requirements.txt` — Python dependencies
- `reports/metrics_summary.csv` — Model evaluation metrics
- `reports/generate_figures.py` — Script to generate result figures
- `reports/figures/` — Output figures and plots

## Installation

1. Clone the repository.
2. Install dependencies:
	```
	pip install -r requirements.txt
	```

## Usage

1. Prepare the dataset in `ML project/data/cardekho.csv`.
2. Run the main script or individual model scripts:
	```
	python "ML project/data/main.py"
	```
	Or run a specific model script, e.g.:
	```
	python "ML project/data/ridgeregression.py"
	```

3. Generate figures:
	```
	python reports/generate_figures.py
	```

## Results

- Model metrics are saved in `reports/metrics_summary.csv`.
- Figures are saved in `reports/figures/`.

## Notes

- The project currently implements Ridge Regression, Decision Tree, XGBoost, and SVM models.
- Make sure all dependencies are installed as per `requirements.txt`.

# Vehicle Price Predictor

A machine learning project to predict the selling price of used cars using regression models (Ridge Regression, Decision Tree, XGBoost, SVM) on the Cardekho dataset. Includes data preprocessing, model evaluation, and result visualization.

## Project Structure

- `ML project/data/cardekho.csv` — Main dataset
- `ML project/data/main.py` — Main script to run experiments
- `ML project/data/decisiontree.py` — Decision Tree model
- `ML project/data/ridgeregression.py` — Ridge Regression model
- `ML project/data/svm.py` — Support Vector Machine model
- `ML project/data/xboost.py` — XGBoost model
- `requirements.txt` — Python dependencies
- `reports/metrics_summary.csv` — Model evaluation metrics
- `reports/generate_figures.py` — Script to generate result figures
- `reports/figures/` — Output figures and plots

## Installation

1. Clone the repository.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Prepare the dataset in `ML project/data/cardekho.csv`.
2. Run the main script or individual model scripts:
   ```
   python "ML project/data/main.py"
   ```
   Or run a specific model script, e.g.:
   ```
   python "ML project/data/ridgeregression.py"
   ```

3. Generate figures:
   ```
   python reports/generate_figures.py
   ```

## Results

- Model metrics are saved in `reports/metrics_summary.csv`.
- Figures are saved in `reports/figures/`.

## Notes

- The project currently implements Ridge Regression, Decision Tree, XGBoost, and SVM models.
- Make sure all dependencies are installed as per `requirements.txt`.
