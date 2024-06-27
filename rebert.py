!pip install datasets
!pip install transformers

import tensorflow as tf
import numpy as np
from datasets import load_dataset, concatenate_datasets, DatasetDict,load_from_disk
import pandas as pd
from transformers import AutoTokenizer,create_optimizer,TFAutoModelForSequenceClassification,DataCollatorWithPadding
from transformers.keras_callbacks import PushToHubCallback, KerasMetricCallback
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from huggingface_hub import notebook_login
from sklearn.preprocessing import LabelEncoder

notebook_login()

"""# loading dataset"""

dataset=load_dataset("relbert/t_rex")

print(dataset["train"][14])

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-cased')

# Function to count tokens in the 'text' column
def count_tokens(example):
    return len(tokenizer(example['text'])['input_ids'])

# Filter out examples with more than 200 tokens
max_token_count = 200
filtered_dataset = dataset.filter(lambda example: count_tokens(example) <= max_token_count)

dataset=filtered_dataset
print(dataset)

print(dataset[0])

# Load train and validation sets
train_data = dataset['train']
validation_data = dataset['validation']

# Concatenate train and validation sets
dataset = concatenate_datasets([train_data, validation_data])

print(dataset)

"""# keep only the examples with the 250 most common relations"""

df = pd.DataFrame(dataset)
relation_counts = df['relation'].value_counts()
sorted_relations = relation_counts.index.tolist()
top_250_relations = sorted_relations[:250]
dataset = df[df['relation'].isin(top_250_relations)]

dataset.info()

"""# remove examples where head and tail are not found in the text"""

df = pd.DataFrame(dataset)
dataset = df[df.apply(lambda row: row['head'] in row['text'] and row['tail'] in row['text'], axis=1)]
# filter out examples where both 'head' and 'tail' are not found in the 'text'

dataset.info()

"""# split into train and validation set with stratified sampling"""

from sklearn.model_selection import train_test_split

# Splitting into train and validation (80% train, 20% validation)
train_data, validation_data = train_test_split(
    dataset,
    test_size=0.2,
    random_state=42,
    stratify=dataset['relation']  # Stratified split based on the 'relation' column
)


# stratify sample a smaller training and validation dataset
from sklearn.model_selection import train_test_split

#stratified sampling on the training data to get a smaller training dataset
smaller_train_data, _ = train_test_split(
    train_data,
    train_size=300000,
    random_state=42,
    stratify=train_data['relation']
)

#stratified sampling on the validation data to get a smaller validation dataset
smaller_validation_data, _ = train_test_split(
    validation_data,
    train_size=60000,
    random_state=42,
    stratify=validation_data['relation']
)

train_data=smaller_train_data
validation_data=smaller_validation_data

"""# add head and tail markers to the text column of each example"""

def add_entity_markers(example):
    # Find entity positions
    head = example['head']
    tail = example['tail']
    # Insert entity markers into text
    text_with_entities = example['text'].replace(head, f"[HEAD]{head}[/HEAD]") \
                         .replace(tail, f"[TAIL]{tail}[/TAIL]")

    # Update the example with modified text
    example['text'] = text_with_entities

    return example

train_df = pd.DataFrame(train_data)
train_df = train_df.apply(add_entity_markers, axis=1)

validation_df = pd.DataFrame(validation_data)
validation_df = validation_df.apply(add_entity_markers, axis=1)

train_df.head()

"""# tokenize dataset"""

train_df.reset_index(drop=True, inplace=True)
validation_df.reset_index(drop=True, inplace=True)

from datasets import Dataset

train_dataset=Dataset.from_pandas(train_df)
validation_dataset=Dataset.from_pandas(validation_df)

print(train_dataset[0])
print(validation_dataset[0])

from sklearn.preprocessing import LabelEncoder
# Create a label encoder instance
label_encoder = LabelEncoder()

# Fit the LabelEncoder on all possible labels (assuming 'relation' is your label column)
all_labels = np.concatenate([train_dataset["relation"], validation_dataset["relation"]])
label_encoder.fit(all_labels)

# Load the BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-cased')

def tokenize_example(example):
    text = example["text"]
    labels = example["relation"]
    tokenized_inputs = tokenizer(text, truncation=True, padding=True,max_length=220)
    example["input_ids"] = tokenized_inputs["input_ids"]
    example["attention_mask"] = tokenized_inputs["attention_mask"]
    example["labels"] = label_encoder.transform(labels)
    return example

tokenized_train_dataset = train_dataset.map(tokenize_example, batched=True)
tokenized_validation_dataset = validation_dataset.map(tokenize_example, batched=True)

# Save the train dataset to Google Drive
tokenized_train_dataset.save_to_disk('/content/drive/MyDrive/tokenized_train_dataset_v4')

# Save the validation dataset to Google Drive
tokenized_validation_dataset.save_to_disk('/content/drive/MyDrive/tokenized_validation_dataset_v4')

"""# loading bert model and tokenizer, setting training, validation dataset and fitting the model"""

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-cased')
model_name = "distilbert-base-cased"
model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=250)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_set = model.prepare_tf_dataset(
    tokenized_train_dataset,
    shuffle=True,
    batch_size=128,
    collate_fn=data_collator,
)

validation_set = model.prepare_tf_dataset(
    tokenized_validation_dataset,
    shuffle=False,
    batch_size=64,
    collate_fn=data_collator,
)

#pushing the model to my huggingface repository
push_to_hub_callback = PushToHubCallback(
    output_dir="rebert_trex_reformed_v5",
    tokenizer=tokenizer,
)

from tensorflow.keras.optimizers import Adam
callbacks = [push_to_hub_callback]
model.compile(optimizer=Adam(3e-5))

model.fit(
    train_set,
    validation_data=validation_set,
    epochs=10,
    batch_size=64,
    callbacks=callbacks
)