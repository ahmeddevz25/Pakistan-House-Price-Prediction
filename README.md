# 🏡 Pakistan House Price Prediction (PakRealEstate AI)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.58.0-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9.0-F7931E.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Machine Learning property valuation application that estimates the fair market value of residential homes across 5 major Pakistani metropolises (**Islamabad, Karachi, Lahore, Rawalpindi, and Faisalabad**). 

The application utilizes an optimized **Random Forest Regressor** pipeline with **92.8% $R^2$ accuracy** and provides an interactive, live **Streamlit Web Application** with confidence intervals, market analytics, and comparable property benchmarks.

---

## 📌 Problem Statement
Real estate buyers, sellers, and agents in Pakistan frequently struggle to ascertain the true market value of properties, often relying on guesswork or subjective broker estimates. This project solves this challenge by predicting property prices (`price_lakh_pkr`) directly from verifiable structural, spatial, and geographic attributes.

---

## 🏗️ 7-Step Solution Pipeline Architecture

1. **Exploratory Data Analysis (EDA)**: Comprehensive analysis of 1,200 verified property records across 25 prime sectors, analyzing distributions, per-marla benchmarks, and amenities premiums.
2. **Data Preprocessing**: Removal of non-predictive `property_id`, separation of target vector $y$, and categorical encoding (`city`, `location`) using Scikit-Learn `OneHotEncoder(handle_unknown='ignore')` inside a `ColumnTransformer`.
3. **Train-Test Split**: Partitioning data into 80% training (960 instances) and 20% testing (240 instances) with a fixed seed (`random_state=42`).
4. **Model Training**: Baseline comparison between **Linear Regression** and an optimized **Random Forest Regressor** (120 estimators, depth 14).
5. **Model Evaluation**: Assessing generalization on unseen test data using **MAE**, **RMSE**, and **$R^2$ Score**.
6. **Model Serialization & Optimization**: Exporting the full inference pipeline using `joblib` compression (`compress=3`), shrinking the artifact by **84% (from 7.5 MB down to 1.2 MB)**.
7. **Streamlit Web Application**: Modern interactive UI with dynamic cascading dropdowns, unit synchronizers, cached Plotly visualizations, and valuation bands.

---

## 📊 Model Benchmark Results

| Model Architecture | Train $R^2$ | Test $R^2$ (Unseen Data) | Test MAE (Lakh PKR) | Test RMSE (Lakh PKR) | Model Size |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression** | 93.43% | 92.42% | ₨ 20.51 L | ₨ 30.33 L | 2 KB |
| **Random Forest Regressor (Selected)** | **98.62%** | **92.80%** | **₨ 18.10 L** | **₨ 29.55 L** | **1.2 MB** |

### 🎯 Key Price Determinants (Feature Importance)
1. **Area in Sq. Feet (`area_sqft`)**: 62.7%
2. **Area in Marla (`area_marla`)**: 10.4%
3. **Islamabad Sector Premium**: 9.8%
4. **Faisalabad Adjustment**: 5.2%
5. **Property Age (`age_years`)**: 3.9%
6. **Bedrooms**: 2.7%

---

## 📁 Repository Structure

```text
├── app.py                      # Complete Streamlit Web Application
├── main.ipynb                  # Fully documented & executed Jupyter Notebook
├── train_model.py              # Reproducible training & pipeline generation script
├── house_price_pakistan.csv    # Dataset (1,200 records, 15 columns)
├── house_price_model.pkl       # Serialized Random Forest pipeline (1.2 MB)
├── linear_regression_model.pkl # Baseline Linear Regression pipeline
├── model_metrics.json          # Evaluation metrics storage
├── feature_importances.csv     # Extracted feature weights
├── city_location_map.json      # Dynamic city-to-sector hierarchy
├── dataset_summary.json        # Precomputed summary stats for UI sliders
├── run_app.bat                 # 1-Click Windows execution script
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 🚀 Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/ahmeddevz25/Pakistan-House-Price-Prediction.git
cd Pakistan-House-Price-Prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Application
Run via Streamlit:
```bash
streamlit run app.py
```
*(On Windows, you can also simply double-click `run_app.bat`)*

The web app will automatically open in your default browser at `http://localhost:8501`.

---

## 🌟 Application Features

- **Live Valuation Calculator**: Dynamic cascading dropdowns (choosing *Lahore* loads DHA, Gulberg, Johar Town, etc.), auto-syncing Marla to Sqft conversion, and instant valuation calculation with 95% confidence bounds ($\pm 18$ Lakh).
- **Market Intelligence & EDA Dashboard**: Interactive Plotly charts for city pricing comparisons, national price distributions, and amenities financial impact.
- **Model Diagnostics**: Transparent 7-step architecture walkthrough, feature importance breakdowns, and metric comparison tables.
- **Dataset Explorer**: Filterable records table with instant CSV export and real estate advice for buyers and sellers.

---

## 👤 Author
Developed by **Ahmed** ([@ahmeddevz25](https://github.com/ahmeddevz25))
