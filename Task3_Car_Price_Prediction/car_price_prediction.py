import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Standalone mock dataset representing vehicle attributes and target valuations
data = {
    'brand': ['Toyota', 'Honda', 'BMW', 'Toyota', 'Ford', 'BMW', 'Honda', 'Ford', 'Toyota', 'BMW'],
    'year': [2018, 2019, 2020, 2015, 2017, 2022, 2016, 2018, 2021, 2019],
    'mileage': [45000, 32000, 15000, 85000, 60000, 5000, 72000, 52000, 18000, 28000],
    'fuel_type': ['Petrol', 'Petrol', 'Diesel', 'Petrol', 'Diesel', 'Petrol', 'Petrol', 'Petrol', 'Petrol', 'Diesel'],
    'price': [18500, 19200, 35000, 12000, 15500, 52000, 13800, 16000, 24500, 31000]
}

df = pd.DataFrame(data)

print("--- Initializing Car Price Regression Pipeline ---")
print(df.head())

# 2. Separate Features and Target Variable
X = df.drop('price', axis=1)
y = df['price']

# 3. Preprocessing: Vectorize categorical variables cleanly via OneHot Encoding
categorical_features = ['brand', 'fuel_type']
numeric_features = ['year', 'mileage']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# 4. Assemble the Scikit-Learn Pipeline with an Ensemble Random Forest Estimator
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# 5. Split data into historical training data and test evaluations
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train the model pipeline 
model_pipeline.fit(X_train, y_train)

# 7. Evaluate Regression Model Performance
y_pred = model_pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Regression Model Evaluation Metrics ---")
print(f"Mean Absolute Error (MAE): ${mae:.2f}")
print(f"R² Predictor Score: {r2:.4f}")

# 8. Unseen feature testing (Custom out-of-sample prediction)
print("\n--- Running Custom Car Valuation Prediction ---")
new_car = pd.DataFrame([{
    'brand': 'Toyota',
    'year': 2020,
    'mileage': 25000,
    'fuel_type': 'Petrol'
}])

predicted_val = model_pipeline.predict(new_car)
print(f"Specs: Toyota, 2020, 25k miles, Petrol -> Predicted Valuation: ${predicted_val[0]:.2f}")