import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load cleaned data
df = pd.read_csv("data/cleaned_data.csv")

# Remove rows with missing text or labels
df = df.dropna(subset=["cleaned_text", "personality_type"])

X = df["cleaned_text"]
y = df["personality_type"]

# Vectorize text
vectorizer = TfidfVectorizer(max_features=1000)
X_vec = vectorizer.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=300)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, "model/personality_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")
print("Model and vectorizer saved.")
print(model.score(X_test,y_test))
