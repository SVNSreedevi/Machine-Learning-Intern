import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("dataset.csv")

# Split data
X = df["text"]
y = df["label"]

# Convert text → features
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Accuracy
print("Training Completed!")
print("Model Accuracy:", model.score(X_test, y_test))

# -------- Prediction Function --------
def predict_sentiment(text):
    text_vec = vectorizer.transform([text])
    result = model.predict(text_vec)[0]
    return result

# -------- Test Input from User --------
while True:
    msg = input("\nEnter a message (or 'exit'): ")
    if msg.lower() == "exit":
        break
    print("Sentiment:", predict_sentiment(msg))
