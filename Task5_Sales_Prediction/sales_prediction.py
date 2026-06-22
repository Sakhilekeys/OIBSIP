import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Standalone dataset representing advertising budgets and corresponding product sales
data = {
    'TV': [230.1, 44.5, 17.2, 151.5, 180.8, 8.7, 57.5, 120.2, 8.6, 199.8],
    'Radio': [37.8, 39.3, 45.9, 41.3, 10.8, 48.9, 32.8, 19.6, 2.1, 2.6],
    'Newspaper': [69.2, 45.1, 69.3, 58.5, 58.4, 75.0, 23.5, 11.6, 1.0, 21.2],
    'Sales': [22.1, 10.4, 9.3, 18.5, 12.9, 7.2, 11.8, 13.2, 4.8, 10.6]
}

df = pd.DataFrame(data)

print("--- Initializing Sales Prediction Regression Pipeline ---")
print(df.head())

# 2. Separate independent feature budgets from the target Sales revenue metric
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# 3. Train/Test Evaluation Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and Train a Multi-Variable Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Evaluate Performance
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Sales Model Evaluation Metrics ---")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R² Predictor Variance Score: {r2:.4f}")

# 6. Run an out-of-sample custom inference budget plan
print("\n--- Simulating Future Campaign Sales Yield ---")
future_campaign = pd.DataFrame([{
    'TV': 150.0,
    'Radio': 25.0,
    'Newspaper': 10.0
}])

predicted_sales = model.predict(future_campaign)
print(f"Proposed Budgets: TV: $150, Radio: $25, News: $10 -> Estimated Revenue Yield: {predicted_sales[0]:.2f} units")