!pip install fastcoref
!pip install transformers
from transformers import pipeline, TFAutoModelForTokenClassification, AutoTokenizer, AutoModelForSequenceClassification
import tensorflow as tf
from huggingface_hub import notebook_login

notebook_login()

from google.colab import drive
drive.mount('/content/drive')

"""# coreference resolution using fastcoref and spacy"""

text="The Defiant Ones is a 1986 American made-for-television crime drama film directed by David Lowell Rich starring Robert Urich and Carl Weathers. It is a remake of the 1958 film of the same name."

print(text)

from fastcoref import spacy_component
import spacy
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("fastcoref")

doc = nlp(
   text,
   component_cfg={"fastcoref": {'resolve_text': True}}
)
text=doc._.resolved_text
print(text)

"""# loading rel model"""

from datasets import load_from_disk
# Load the train dataset from Google Drive
tokenized_train_dataset = load_from_disk('/content/drive/MyDrive/tokenized_train_dataset_v4')

# Load the validation dataset from Google Drive
tokenized_validation_dataset = load_from_disk('/content/drive/MyDrive/tokenized_validation_dataset_v4')

from sklearn.preprocessing import LabelEncoder
import numpy as np
# Create a label encoder instance
label_encoder = LabelEncoder()
# Combine train and validation labels to get all possible labels
all_labels = np.concatenate([tokenized_train_dataset["relation"],tokenized_validation_dataset["relation"]])

# Fit the LabelEncoder on all possible labels
label_encoder.fit(all_labels)

import json
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import spacy
from itertools import combinations

# Load the fine-tuned model
model = TFAutoModelForSequenceClassification.from_pretrained('nikoslefkos/rebert_v3')

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('nikoslefkos/rebert_v3')

# Load the PID to name mapping from the pid2name.json file
with open('/content/drive/MyDrive/pid2name_v3.json', 'r') as f:
    pid2name = json.load(f)

ner_classifier=pipeline("ner",model="nikoslefkos/nerbert_ontonotes",grouped_entities=True)


"""# sentencewise relation extraction"""

import spacy
from itertools import combinations

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Process the text
doc = nlp(text)
triples = []
unique_triples= []

for sentence in doc.sents:
    sentence_text = sentence.text
    entities = ner_classifier(sentence_text)
    entities = [entity for entity in entities if entity['score'] >= 0.9]

    # Create a set to keep track of unique words
    unique_words = set()

    # Create a new list to store unique entities
    unique_entities = []

    # Iterate through the list of entities
    for entity in entities:
        word = entity['word']
        if word not in unique_words:
            unique_words.add(word)
            unique_entities.append(entity)

    entity_pairs = list(combinations(unique_entities, 2))

    for entity_pair in entity_pairs:
        modified_sentence = sentence_text
        head_entity = entity_pair[0]['word']
        tail_entity = entity_pair[1]['word']
        modified_sentence = modified_sentence.replace(head_entity, f"[HEAD]{head_entity}[/HEAD]") \
                         .replace(tail_entity, f"[TAIL]{tail_entity}[/TAIL]")
        # Tokenize the sentence
        input = tokenizer(modified_sentence, truncation=True, padding=True, max_length=200, return_tensors="tf")

        # Make predictions
        sentence_prediction = model(input)

        logits = sentence_prediction.logits.numpy()
        probabilities = tf.nn.softmax(logits, axis=-1).numpy()

        label_id = np.argmax(probabilities, axis=1)
        predicted_label = label_encoder.inverse_transform(label_id)
        predicted_label = predicted_label[0]

        confidence_score = np.max(probabilities, axis=1)


        #if predicted_label in pid2name:
            #predicted_label = pid2name[str(predicted_label)]

        if confidence_score >= 0.7:
            triples.append((head_entity, predicted_label, tail_entity))

for tuple in triples:
  if tuple not in unique_triples:
    unique_triples.append(tuple)

for triple in unique_triples:
  print(triple)

folder_path = '/content/drive/MyDrive'

with open(f'{folder_path}/triples.txt', 'w') as f:
    for item in unique_triples:
        f.write(f'{item}\n')
