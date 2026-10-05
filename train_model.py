import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def main():
    print("=== STEP 1: Loading Dataset ===")
    df = pd.read_csv('house_price_pakistan.csv')
    print(f"Total rows: {len(df)}, columns: {len(df.columns)}")

    # Extract City to Location Mapping
    city_location_map = {}
    for city in sorted(df['city'].unique()):
        locs = sorted(df[df['city'] == city]['location'].unique().tolist())
        city_location_map[city] = locs
    
    with open('city_location_map.json', 'w') as f:
        json.dump(city_location_map, f, indent=4)
    print("Saved city_location_map.json")

    # Dataset Summary Statistics for UI Sliders / Ranges
    summary = {
        'num_records': len(df),
        'cities': sorted(df['city'].unique().tolist()),
        'area_marla_min': int(df['area_marla'].min()),
        'area_marla_max': int(df['area_marla'].max()),
        'area_marla_default': int(df['area_marla'].median()),
        'area_sqft_min': int(df['area_sqft'].min()),
        'area_sqft_max': int(df['area_sqft'].max()),
        'bedrooms_min': int(df['bedrooms'].min()),
        'bedrooms_max': int(df['bedrooms'].max()),
        'bedrooms_default': int(df['bedrooms'].median()),
        'bathrooms_min': int(df['bathrooms'].min()),
        'bathrooms_max': int(df['bathrooms'].max()),
        'bathrooms_default': int(df['bathrooms'].median()),
        'stories_options': sorted(df['stories'].unique().tolist()),
        'age_years_min': int(df['age_years'].min()),
        'age_years_max': int(df['age_years'].max()),
        'age_years_default': int(df['age_years'].median()),
        'school_km_min': float(df['nearby_school_km'].min()),
        'school_km_max': float(df['nearby_school_km'].max()),
        'school_km_default': round(float(df['nearby_school_km'].median()), 2),
        'price_lakh_min': float(df['price_lakh_pkr'].min()),
        'price_lakh_max': float(df['price_lakh_pkr'].max()),
        'price_lakh_mean': round(float(df['price_lakh_pkr'].mean()), 2),
        'avg_sqft_per_marla': round(float((df['area_sqft'] / df['area_marla']).mean()), 1)
    }
    with open('dataset_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)
    print("Saved dataset_summary.json")

    # === STEP 2: Preprocessing ===
    # Drop property_id as specified in House_Price_Prediction.docx
    X = df.drop(columns=['property_id', 'price_lakh_pkr'])
    y = df['price_lakh_pkr']

    categorical_cols = ['city', 'location']
    numerical_cols = [c for c in X.columns if c not in categorical_cols]

    # Preprocessor using OneHotEncoder with handle_unknown='ignore'
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_cols)
        ],
        remainder='passthrough'
    )

    # === STEP 3: Train-Test Split (80-20) ===
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    # === STEP 4: Train Models ===
    # 1. Linear Regression Baseline
    lr_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    lr_pipeline.fit(X_train, y_train)

    # 2. Optimized Random Forest Regressor
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=120,
            max_depth=14,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ])
    rf_pipeline.fit(X_train, y_train)

    # === STEP 5: Evaluate Models ===
    def get_metrics(model, X_tr, y_tr, X_te, y_te):
        pred_tr = model.predict(X_tr)
        pred_te = model.predict(X_te)
        return {
            'train_mae': round(float(mean_absolute_error(y_tr, pred_tr)), 2),
            'train_rmse': round(float(np.sqrt(mean_squared_error(y_tr, pred_tr))), 2),
            'train_r2': round(float(r2_score(y_tr, pred_tr)), 4),
            'test_mae': round(float(mean_absolute_error(y_te, pred_te)), 2),
            'test_rmse': round(float(np.sqrt(mean_squared_error(y_te, pred_te))), 2),
            'test_r2': round(float(r2_score(y_te, pred_te)), 4)
        }

    lr_metrics = get_metrics(lr_pipeline, X_train, y_train, X_test, y_test)
    rf_metrics = get_metrics(rf_pipeline, X_train, y_train, X_test, y_test)

    metrics = {
        'Linear Regression': lr_metrics,
        'Random Forest Regressor': rf_metrics
    }
    with open('model_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    print("Model Metrics:")
    print(json.dumps(metrics, indent=2))

    # Feature Importance Extraction
    cat_encoder = rf_pipeline.named_steps['preprocessor'].named_transformers_['cat']
    encoded_cat_names = cat_encoder.get_feature_names_out(categorical_cols).tolist()
    all_feature_names = encoded_cat_names + numerical_cols
    importances = rf_pipeline.named_steps['regressor'].feature_importances_

    feat_df = pd.DataFrame({
        'feature': all_feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    feat_df.to_csv('feature_importances.csv', index=False)
    print("\nTop 10 Feature Importances:")
    print(feat_df.head(10))

    # === STEP 6: Save Models with compression (84% size reduction) ===
    joblib.dump(rf_pipeline, 'house_price_model.pkl', compress=3)
    joblib.dump(lr_pipeline, 'linear_regression_model.pkl', compress=3)
    print("\nSaved compressed house_price_model.pkl and linear_regression_model.pkl")

    # Quick test prediction
    sample_input = pd.DataFrame([{
        'city': 'Lahore',
        'location': 'DHA',
        'area_marla': 10,
        'area_sqft': 2250,
        'bedrooms': 4,
        'bathrooms': 4,
        'stories': 2.0,
        'age_years': 5,
        'main_road_access': 1,
        'garage': 1,
        'gas_available': 1,
        'furnished': 1,
        'nearby_school_km': 0.8
    }])
    sample_pred = rf_pipeline.predict(sample_input)[0]
    print(f"\nTest prediction for 10 Marla in Lahore DHA: {sample_pred:.2f} Lakh PKR ({sample_pred/100:.2f} Crore PKR)")

if __name__ == '__main__':
    main()
