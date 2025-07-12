import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Load your dataset
df = pd.read_csv("data/data.csv")  # Adjust path if needed

# Rename columns for clarity (optional)
df = df.rename(columns={"type": "personality_type", "posts": "raw_posts"})

# Combine all post segments into one cleaned text per row
def clean_text(text):
    if pd.isnull(text):
        return ""
    text = text.replace('|||', ' ')  # Replace delimiters
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # Remove URLs
    text = re.sub(r'@\w+|#\w+', '', text)  # Remove mentions/hashtags
    text = re.sub(r'[^A-Za-z\s]', '', text)  # Remove non-alphabetic
    text = text.lower()
    words = text.split()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Apply cleaning
df["cleaned_text"] = df["raw_posts"].apply(clean_text)

# Save cleaned CSV
df.to_csv("data/cleaned_data.csv", index=False)
print("Cleaned data saved to data/cleaned_data.csv")
