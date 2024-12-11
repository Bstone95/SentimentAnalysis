# %%
import pandas as pd
import numpy as np
df = pd.read_csv('YourCSV') 

df

# %%
df = df.sample(20)
df.columns


# %%
from transformers import pipeline

pipe = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions")

from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")

model = AutoModelForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions")
nlp1 = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

texts = list(df['Comment Body'].dropna().values)
device = 'cuda'

results = nlp1(texts)
for text, result, in zip(texts, results):
    print('Text:', text)
    print('Result:', result)

# %%
from transformers import pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import TrainingArguments, Trainer
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')


nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

texts = list(df['Comment Body'].dropna().values)

train_text, val_text, train_labels, val_labels = train_test_split(texts, test_size=0.2, stratify=0)


device = 'cuda'

results = nlp(texts)

for text, result, in zip(texts, results):
    print('Text:', text)
    print('Result:', result)
    


