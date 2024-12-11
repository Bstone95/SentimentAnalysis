import pandas as pd
import re
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import download
import emoji

# Download NLTK resources
import os



download('punkt')
download('wordnet')
download('stopwords')

# Initialize the stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if pd.isna(text):
        return text
    
    # Remove HTML links
    text = BeautifulSoup(text, "html.parser").text

    # Lowercase
    text = text.lower()

    # Remove punctuations
    text = re.sub(r'[^\w\s]', '', text)

    # Convert emoticons to words
    text = emoji.demojize(text)  # Convert emoticons to corresponding words

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]

    # Stemming and Lemmatization
    tokens = [lemmatizer.lemmatize(stemmer.stem(word)) for word in tokens]

    return ' '.join(tokens)

def preprocess_dataframe(df):
    for column in df.columns:
        if df[column].dtype == object:  # Apply preprocessing only to text columns
            df[column] = df[column].apply(preprocess_text)
    return df

def preprocess_csv(input_csv, output_csv):
    # Read the CSV file
    df = pd.read_csv(input_csv)

    # Preprocess the DataFrame
    df = preprocess_dataframe(df)

    # Save the preprocessed data to a new CSV file
    df.to_csv(output_csv, index=False)

# Example usage
input_csv = r'yourcsv'
output_csv = r'outputcsv'
preprocess_csv(input_csv, output_csv)
print("Current working directory:", os.getcwd())