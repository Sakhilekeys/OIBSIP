import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Create a synthetic corpus to ensure the script runs standalone flawlessly
data = {
    'text': [
        'Hey, are we still meeting up for lunch today?',
        'WINNER! As a valued network customer you have been selected to receive a £900 prize reward! Claim now.',
        'Can you send me the engineering laboratory report updates when you finish?',
        'URGENT! Your mobile number has won a £2,000 bonus prize. Call 09061701461 to claim your cash.',
        'Just checking in to see how your exam preparation is going.',
        'FREE entry into our weekly competition to win a brand new smartphone! Text GO to 87121.',
        'Please review the attached project documentation before tomorrow morning.',
        'Get rich quick! Earn thousands of dollars working from home with no experience needed! Click here.',
        'Are you coming to the study group session at the library later?',
        'Congratulations! You have been awarded a free holiday voucher. Click the link to claim.'
    ],
    'label': ['ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam']
}

# 2. Load into DataFrame
df = pd.DataFrame(data)

print("--- Initializing Data Pipeline ---")
print(df.head())
print("\nTarget Class Distribution:")
print(df['label'].value_counts())

# 3. Feature Extraction (Convert raw text to numeric TF-IDF vectors)
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['text'])
y = df['label']

# 4. Train/Test Split (Fixed the parameter typo here)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 5. Initialize and Train the Model
model = MultinomialNB()
model.fit(X_train, y_train)

# 6. Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n--- Model Evaluation Metrics ---")
print(f"Classification Accuracy: {accuracy * 100:.2f}%")
print("\nDetailed Performance Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# 7. Testing custom user input predictions
print("\n--- Running Custom Inference Tests ---")
test_samples = [
    "Hey friend, are you free to study for the upcoming signals test?",
    "CONGRATULATIONS! You won a cash reward text back now to claim"
]

transformed_samples = vectorizer.transform(test_samples)
predictions = model.predict(transformed_samples)

for message, category in zip(test_samples, predictions):
    print(f"Message: '{message}' -> Classified as: [{category.upper()}]")