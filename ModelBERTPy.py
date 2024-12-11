# %%
#test run to ensure LLM is working properly
%pip install transformers
%pip install torch
%pip install numpy
%pip install pandas

# %%
import pandas as pd

df = pd.read_csv('YourCSV')

df

# %%
df = df.sample(20)
df.columns

# %%
from transformers import pipeline
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')

nlp = pipeline('sentiment-analysis', model = model, tokenizer = tokenizer)

texts = list(df['Comment Body'].dropna().values)

results = nlp(texts)



