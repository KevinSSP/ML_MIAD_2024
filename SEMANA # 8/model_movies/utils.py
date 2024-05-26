# utils.py
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')

def split_into_lemmas(text):
    wordnet_lemmatizer = WordNetLemmatizer()
    text = text.lower()
    words = text.split()
    return [wordnet_lemmatizer.lemmatize(word) for word in words]
