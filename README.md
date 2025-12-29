# Car Price Prediction Project

A machine learning project to predict the selling price of used cars using regression models (Ridge, Random Forest, XGBoost, SVR) on the Cardekho dataset. Includes data preprocessing, model evaluation, and result visualization.

# Vehicle Price Predictor

## Table of Contents


## Project Motivation
The goal of this project is to build robust machine learning models to predict the selling price of used cars based on their features. Accurate price prediction is valuable for both buyers and sellers in the automotive market, enabling fair valuation and better negotiation.

## Dataset Description

### Feature List
| Feature         | Description                                 |
|----------------|---------------------------------------------|
| name           | Car name/model (dropped in modeling)        |
| year           | Year of manufacture                         |
| selling_price  | Target variable (car price)                 |
| km_driven      | Kilometers driven                           |
| fuel           | Fuel type                                   |
| seller_type    | Type of seller (individual/dealer/trust)    |
| transmission   | Transmission type (manual/automatic)        |
| owner          | Ownership status (first, second, etc.)      |
| mileage        | Mileage (km/ltr/kg)                         |
| engine         | Engine displacement (cc)                    |
| max_power      | Maximum power (bhp)                         |
| seats          | Number of seats                             |

**Note:** The `name` column was dropped due to high cardinality and risk of overfitting.

## Preprocessing & Feature Engineering

## Modeling Approach
The following supervised regression models were implemented and compared:
