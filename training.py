import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from sentence_transformers import SentenceTransformer
import joblib

# Optional: Use SMOTE for class balancing
from imblearn.over_sampling import SMOTE

# Load cleaned data
df = pd.read_csv("data/cleaned_data.csv")

# Encode MBTI labels
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["personality_type"])
joblib.dump(label_encoder, "model/label_encoder.pkl")

# Load SentenceTransformer for BERT-like embeddings
bert = SentenceTransformer('all-MiniLM-L6-v2')
print("🔄 Encoding text using BERT...")
X = bert.encode(df["cleaned_text"].tolist(), show_progress_bar=True)
y = df["label"]

# (Optional) Balance classes using SMOTE
print("⚖️ Balancing classes with SMOTE...")
smote = SMOTE()
X, y, *_ = smote.fit_resample(X, y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train XGBoost classifier
print("🚀 Training XGBoost model...")
model = XGBClassifier(objective='multi:softprob', eval_metric='mlogloss')
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\n✅ Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print("✅ Accuracy:", accuracy_score(y_test, y_pred))

# Save the model
joblib.dump(model, "model/personality_model.pkl")
print("✅ Model saved to model/personality_model.pkl")
