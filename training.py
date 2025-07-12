import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib

# Load cleaned data
df = pd.read_csv("data/cleaned_data.csv")
df = df.dropna(subset=["cleaned_text", "personality_type"])

# Encode labels (e.g., ENFP -> 0, INFP -> 1, etc.)
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["personality_type"])

# Save label encoder for decoding later
joblib.dump(label_encoder, "model/label_encoder.pkl")

# Prepare features and labels
X = df["cleaned_text"]
y = df["label"]

# Vectorize text
vectorizer = TfidfVectorizer(max_features=3000)
X_vec = vectorizer.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42, stratify=y
)

# Train XGBoost model
model = XGBClassifier(eval_metric='mlogloss')
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("✅ Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print("✅ Accuracy:", accuracy_score(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, "model/personality_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")
print("✅ Model, vectorizer, and label encoder saved.")
