# %%

import pandas as pd
import numpy as np
from transformers import Trainer, TrainingArguments, RobertaForSequenceClassification, RobertaTokenizer
from sklearn.model_selection import train_test_split
from transformers import DataCollatorWithPadding
import torch

from datasets import Dataset
df = pd.read_csv('YourCSV') #insert personal CSV

df
import transformers

print(torch.cuda.is_available()) #checks for avaialable GPU


# %%
from transformers import pipeline

from transformers import RobertaForSequenceClassification




pipe = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions")

from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")

model = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=28)
nlp1 = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, device = 0)

texts = list(df['Comment Body'].dropna().values)
labels = list(df['Predicted Label'].dropna().values)

df_filtered = df.dropna(subset=['Comment Body', 'Predicted Label'])
device = 'cuda'

results = nlp1(texts, truncation = True)
for text, result, in zip(texts, results):
    print('Text:', text)
    print('Result:', result)

# %%
from datasets import Dataset, DatasetDict

# Rename columns before splitting
df_filtered.rename(columns={'Comment Body': 'text', 'Predicted Label': 'label'}, inplace=True)

# Create a DataFrame for training and validation
train_df = df_filtered.sample(frac=0.8, random_state=42)  # 80% for training
val_df = df_filtered.drop(train_df.index)  # Remaining 20% for validation

# Convert DataFrames to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
val_dataset = Dataset.from_pandas(val_df[['text', 'label']])

# Print the columns to verify they are correct
print(train_df.columns)
print(val_df.columns)

# %%
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    accuracy = accuracy_score(labels, predictions)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# %%
from datasets import Dataset, DatasetDict
label_mapping = {'admiration': 0, 'amusement': 1, 'anger': 2, 'annoyance': 3, 'approval': 4, 'caring': 5,
                 'confusion': 6, 'curiosity': 7,  'desire': 8, 'disappointment': 9, 'disapproval': 10, 'disgust': 11, 'embarrassment': 12, 'excitement': 13, 'fear': 14, 
                 'gratitude': 15, 'grief': 16, 'joy': 17, 'love': 18, 'nervousness': 19, 'optimism': 20, 'pride': 21, 'realization': 22, 'relief': 23, 'remorse': 24, 'sadness': 25,
                 'surprise': 26, 'neutral': 27} #creates labels for emotions

train_df['label'] = train_df['label'].map(label_mapping)
val_df['label'] = val_df['label'].map(label_mapping)

train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
val_dataset = Dataset.from_pandas(val_df[['text', 'label']])



def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)


# Set the format
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])


# %%
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    remove_unused_columns=True,
    # Ensure GPU is used
)

# Initialize Trainer
trainer = Trainer(
    model=model,  # Ensure your model is loaded correctly
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)



# %%
trainer.train()

# %%
eval_results = trainer.evaluate()
print(eval_results)

# %%
trainer.save_model('./finetuned1-SamLowebert')
tokenizer.save_pretrained('./finetuned1-SamLowebert')


