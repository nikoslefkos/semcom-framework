# generating text from triples using Flan-T5

!pip install transformers
!pip install sentencepiece

from transformers import T5Tokenizer, T5ForConditionalGeneration
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

from google.colab import drive
drive.mount('/content/drive')

import json as json
with open('/content/drive/MyDrive/pid2name_v3.json', 'r') as f:
    pid2name = json.load(f)

file_path = '/content/drive/MyDrive/triples.txt'
triples = []
with open(file_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        # Convert the string representation of tuples to actual tuples
        triple = eval(line.strip())
        triples.append(triple)

for triple in triples:
  print(triple)

print(triples)

"""# pid2name mapping"""

updated_triples=[(t[0],pid2name[t[1]] if t[1] in pid2name else t[1],t[2])for t in triples]

for triple in updated_triples:
  print(triple)

"""#function for triples linearization"""

def transform_triple(triple):
    subject, predicate, obj = triple
    if predicate in pid2name:
        predicate = pid2name[predicate]
    transformed_triple = f"<SUBJECT> {subject} <PREDICATE> {predicate} <OBJECT> {obj}"
    return transformed_triple

# Apply transformation
linearized_triples = [transform_triple(triple) for triple in triples]

# Concatenate all transformed triples into a single line
linearized_triples = " ".join(linearized_triples)

# Print the result
print(linearized_triples)

"""# prompting the llm for text generation from given triples"""

prompt = "translate the following triples into text." \
f"Triples:{updated_triples}"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
outputs = model.generate(input_ids, max_length=2500)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)