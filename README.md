# Semantic communications framework with transformer based models
 
 This is a repository for my diploma thesis which presents the creation of an end-to-end semantic
communications framework for sending and receiving text data. The creation of this framework involves the implementation of models that undertake the extraction of semantic information of a text and the usage of a model to reconstruct semantic information. 

## Semantic information extraction

 In order to achieve the extraction of the semantic information of a text, I implemented the following models:
 
 1. Named Entity Recognition

 A NER model was implemented by fine tuning the DistilBERT model on the [OntoNotes dataset](https://huggingface.co/datasets/SpeedOfMagic/ontonotes_english) for token classification. That is shown at [nerbert.py]().
 
 2. Relation Classification

 A relation classification (or relation extraction) model was implemented by fine tuning the DistilBERT model on the [T-Rex dataset](https://huggingface.co/datasets/relbert/t_rex) for sequence classification. That is shown at [rebert.py]().

 The models are then used sequentially to extract semantic information following the pipeline approach. 

 The detailed steps followed to extract semantic information are  the following;
 
 1. Coreference resolution
 2. Separating text into sentences
 3. NER for each sentence
 4. Creating entity pairs
 5. Entity tagging using head and tail markers
 6. Relation classification for each entity pair
 7. Semantic triple creation
 8. Adding semantic triple inside a list containing all the semantic triples of the text


 ##  Semantic information reconstruction

 The receiver generates coherent text based on the semantic triples he received. 
 
 This is achieved through prompting the large language model flan-t5. The prompt that is used is: "Translate the following triples into text: (list of triples)".

 ## How to use the framework

 1. Perform semantic information extraction with triples_extraction.py 
 2. Perform semantic information reconstruction with triples_to_text_llm.py 

 The following video demonstrates the usage of the framework for sending and receiving text data: 
 
 [Demo](https://youtu.be/kkoRAqLQDZU)

 

 ## License

 See the [LICENSE]() file for license rights and limitations.



 




