import pandas as pd

# Load the dataset
df = pd.read_csv('Iris.csv')

# Print the first 5 rows to make sure it loaded correctly
print("Data successfully loaded!")
print(df.head())
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Separate features (X) and target species (y)
X = df.drop(columns=['Id', 'Species'])
y = df['Species']

# 2. Split into 80% training and 20% testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize and train the model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 4. Make predictions and print the final accuracy score
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Model Training Complete!")
print(f"Accuracy Score: {accuracy * 100:.2f}%")