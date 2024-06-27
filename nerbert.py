!pip install datasets
!pip install transformers
!pip install evaluate
!pip install seqeval

from huggingface_hub import notebook_login
notebook_login()

import tensorflow as tf
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer,DataCollatorForTokenClassification,create_optimizer,TFAutoModelForTokenClassification,DistilBertTokenizer,pipeline
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers.keras_callbacks import PushToHubCallback, KerasMetricCallback
from huggingface_hub import notebook_login
seqeval = evaluate.load("seqeval")

dataset=load_dataset("SpeedOfMagic/ontonotes_english")

print(dataset)

print(dataset["train"][0])

dataset=dataset.rename_column("ner_tags","labels")
dataset=dataset.rename_column("tokens","words")

#load the bert model and its tokenizer
model_checkpoint="distilbert-base-cased"
tokenizer=AutoTokenizer.from_pretrained(model_checkpoint)

def shift_label(label):
  if label % 2 == 1:  #if the label is B-XXX we change it to I-XXX(used for subword tokens)
    label+=1
  return label

#aligns the labels with the corresponding tokenized inputs
def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id is None:
            new_labels.append(-100)
        elif word_id != current_word:
            current_word = word_id
            new_labels.append(labels[word_id])
        else:
            new_labels.append(shift_label(labels[word_id]))
    return new_labels

#tokenizes the input examples and aligns the labels with the tokenized inputs
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["words"], truncation=True, is_split_into_words=True)
    new_labels = []
    for i, labels in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

#tokenize and align the labels for each example in the dataset
tokenized_dataset=dataset.map(tokenize_and_align_labels,batched=True)

# batch, pad, and collate the tokenized inputs and labels into a format that can be fed into the model
data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer,return_tensors="tf")

#map label indices to their corresponding labels and vice versa
id2label = {
    0: "O",
    1: "B-PERSON",
    2: "I-PERSON",
    3: "B-NORP",
    4: "I-NORP",
    5: "B-FAC",
    6: "I-FAC",
    7: "B-ORG",
    8: "I-ORG",
    9: "B-GPE",
    10: "I-GPE",
    11: "B-LOC",
    12: "I-LOC",
    13: "B-PRODUCT",
    14: "I-PRODUCT",
    15: "B-DATE",
    16: "I-DATE",
    17: "B-TIME",
    18: "I-TIME",
    19: "B-PERCENT",
    20: "I-PERCENT",
    21: "B-MONEY",
    22: "I-MONEY",
    23: "B-QUANTITY",
    24: "I-QUANTITY",
    25: "B-ORDINAL",
    26: "I-ORDINAL",
    27: "B-CARDINAL",
    28: "I-CARDINAL",
    29: "B-EVENT",
    30: "I-EVENT",
    31: "B-WORK_OF_ART",
    32: "I-WORK_OF_ART",
    33: "B-LAW",
    34: "I-LAW",
    35: "B-LANGUAGE",
    36: "I-LANGUAGE"
}

label2id ={
    "O": 0,
    "B-PERSON": 1,
    "I-PERSON": 2,
    "B-NORP": 3,
    "I-NORP": 4,
    "B-FAC": 5,
    "I-FAC": 6,
    "B-ORG": 7,
    "I-ORG": 8,
    "B-GPE": 9,
    "I-GPE": 10,
    "B-LOC": 11,
    "I-LOC": 12,
    "B-PRODUCT": 13,
    "I-PRODUCT": 14,
    "B-DATE": 15,
    "I-DATE": 16,
     "B-TIME": 17,
    "I-TIME": 18,
    "B-PERCENT": 19,
    "I-PERCENT": 20,
    "B-MONEY": 21,
    "I-MONEY": 22,
    "B-QUANTITY": 23,
    "I-QUANTITY": 24,
    "B-ORDINAL": 25,
    "I-ORDINAL": 26,
   "B-CARDINAL": 27,
    "I-CARDINAL": 28,
    "B-EVENT": 29,
    "I-EVENT": 30,
    "B-WORK_OF_ART": 31,
    "I-WORK_OF_ART": 32,
    "B-LAW": 33,
    "I-LAW": 34,
    "B-LANGUAGE": 35,
    "I-LANGUAGE": 36
}

#setting the hyperparameters of the model
batch_size=32
num_train_epochs=10
num_train_steps = (len(tokenized_dataset["train"]) // batch_size) * num_train_epochs
optimizer, lr_schedule = create_optimizer(
    init_lr=3e-5,
    num_train_steps=num_train_steps,
    weight_decay_rate=0.01,
    num_warmup_steps=0,
)

#fine tune the distilbert-base-cased model for token classifaction on the conll03 dataset
model = TFAutoModelForTokenClassification.from_pretrained(
    "distilbert-base-cased", num_labels=37, id2label=id2label, label2id=label2id
)

label_list=[
        "O",
    "B-CARDINAL",
    "B-DATE",
    "I-DATE",
    "B-PERSON",
    "I-PERSON",
    "B-NORP",
    "B-GPE",
    "I-GPE",
    "B-LAW",
    "I-LAW",
    "B-ORG",
    "I-ORG",
    "B-PERCENT",
    "I-PERCENT",
    "B-ORDINAL",
    "B-MONEY",
    "I-MONEY",
    "B-WORK_OF_ART",
    "I-WORK_OF_ART",
    "B-FAC",
    "B-TIME",
    "I-CARDINAL",
    "B-LOC",
    "B-QUANTITY",
    "I-QUANTITY",
    "I-NORP",
    "I-LOC",
    "B-PRODUCT",
    "I-TIME",
    "B-EVENT",
    "I-EVENT",
    "I-FAC",
    "B-LANGUAGE",
    "I-PRODUCT",
    "I-ORDINAL",
    "I-LANGUAGE"
]
labels=label_list
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

#create train and validation tensorflow datasets
train_set = model.prepare_tf_dataset(
    tokenized_dataset["train"],
    shuffle=True,
    batch_size=32,
    collate_fn=data_collator,
)

validation_set = model.prepare_tf_dataset(
    tokenized_dataset["validation"],
    shuffle=False,
    batch_size=32,
    collate_fn=data_collator,
)

#pushing the model to my huggingface repository
push_to_hub_callback = PushToHubCallback(
    output_dir="nerbert_ontonotes",
    tokenizer=tokenizer,
)

from transformers.keras_callbacks import KerasMetricCallback
metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=validation_set)

model.compile(optimizer=optimizer)

#specifying the callbacks and fitting the model
callbacks = [push_to_hub_callback,metric_callback]
model.fit(x=train_set, validation_data=validation_set, epochs=10,callbacks=callbacks)

text="My name is Nikolaos Lefkos and i live in Lamia,Greece. I am a  student at the University of Western Macedonia"
ner_classifier=pipeline("ner",model="nikoslefkos/nerbert_ontonotes",grouped_entities=True)
ner_classifier(text)