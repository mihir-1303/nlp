
import re
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample supply-chain news
news = [
    "Severe flooding has closed the main shipping port.",
    "Workers announced a strike at the manufacturing facility.",
    "The supplier reported normal production and on-time delivery.",
    "Heavy rainfall caused delays in cargo transportation."
]

# Basic text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

cleaned_news = [clean_text(text) for text in news]

# Convert text into numerical features
vectorizer = TfidfVectorizer(stop_words="english")
features = vectorizer.fit_transform(cleaned_news)

print("NLP preprocessing successful!")
print("\nCleaned news:")
for text in cleaned_news:
    print("-", text)

print("\nTF-IDF matrix shape:", features.shape)
print("\nExtracted vocabulary:")
print(vectorizer.get_feature_names_out())

# Save the processed dataset
df = pd.DataFrame({
    "original_text": news,
    "cleaned_text": cleaned_news
})

df.to_csv("nlp_test_output.csv", index=False)
print("\nSaved nlp_test_output.csv")
