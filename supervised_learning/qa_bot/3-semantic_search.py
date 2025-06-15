#!/usr/bin/env python3
"""QA with BertTokenizer and bert-uncased-tf2-qa model"""


import tensorflow_hub as hub
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
import os


def semantic_search(corpus_path, sentence):
    """Performs semantic search on corpus of documents"""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = TFBertModel.from_pretrained('bert-base-uncased')
    
    files = []
    references = []
    corpus_content = os.listdir(corpus_path)
    for file in corpus_content:
        file_path = os.path.join(corpus_path, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            references.append(f.read())
            files.append(file)
            
    tokenized_reference = tokenizer(references, return_tensors='tf', padding=True, truncation=True)
    outputs = model(tokenized_reference)
    embedding = outputs.last_hidden_state.mean(axis=1)
    reference_embeddings.append(embedding)
    reference_embeddings = tf.concat(reference_embeddings, axis=0).numpy()
    
    tokenized_sentence = tokenizer(sentence, return_tensors='tf', padding=True, truncation=True)
    sentence_outputs = model(tokenized_sentence)
    sentence_embedding = sentence_outputs.last_hidden_state.mean(axis=1).numpy()
    
    cos_similarity = cosine_similarity(sentence_embedding, reference_embeddings)
    most_similar_reference_idx = cos_similarity.argmax()
    most_similar_reference = os.path.join(corpus_path, files[most_similar_reference_idx])
    return most_similar_reference
