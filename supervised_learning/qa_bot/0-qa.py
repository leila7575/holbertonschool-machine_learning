#!/usr/bin/env python3
"""QA with BertTokenizer and bert-uncased-tf2-qa model"""


import tensorflow_hub as hub
from transformers import BertTokenizer
import tensorflow as tf


def question_answer(question, reference):
    """finds a snippet of text within a reference document to answer a question"""
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    model = hub.load('https://www.kaggle.com/models/seesee/bert/TensorFlow2/uncased-tf2-qa/1')
    tokenized_question = tokenizer.tokenize(question)
    tokenized_reference = tokenizer.tokenize(reference)
    tokens = ['[CLS]'] + tokenized_question + ['[SEP]'] + tokenized_reference + ['[SEP]']
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    mask = [1] * len(input_ids)
    segment_type_ids = [0] * (1 + len(tokenized_question) + 1) + [1] * (len(tokenized_reference) + 1)
    input_ids = tf.convert_to_tensor([input_ids], dtype=tf.int32)
    mask = tf.convert_to_tensor([mask], dtype=tf.int32)
    segment_type_ids = tf.convert_to_tensor([segment_type_ids], dtype=tf.int32)
    outputs = model([input_ids, mask, segment_type_ids])
    start_token_index = tf.argmax(outputs[0][0][1:]).numpy() + 1
    end_token_index = tf.argmax(outputs[1][0][1:]).numpy() + 1
    if end_token_index < start_token_index:
        return None
    answer = tokenizer.convert_tokens_to_string(tokens[start_token_index: end_token_index + 1])
    return answer
