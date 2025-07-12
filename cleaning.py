import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources if not already present
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
remove_words = set(stopwords.words("english"))

# Define the clean_text function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+', ' ', text)         # Remove URLs
    text = re.sub(r'[0-9]', ' ', text)                # Remove numbers
    text = re.sub(r'\W+', ' ', text)                  # Remove non-alphanumeric
    text = re.sub(r'[_+]', ' ', text)                 # Remove underscores/pluses
    text = re.sub(r'\s+', ' ', text)                  # Remove extra spaces
    words = [w for w in text.split() if w not in remove_words]
    lemmatized = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(lemmatized)

# Load dataset
df = pd.read_csv("data/data.csv")
df = df.rename(columns={"type": "personality_type", "posts": "raw_posts"})

# Clean posts
df["cleaned_text"] = df["raw_posts"].apply(clean_text)

# Save cleaned data
df.to_csv("data/cleaned_data.csv", index=False)
print("✅ Cleaned data saved to data/cleaned_data.csv")
