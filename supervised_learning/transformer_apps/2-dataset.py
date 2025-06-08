#!/usr/bin/env python3
"""Class dataset, loading dataset for machine translation"""


import transformers
import tensorflow_datasets as tfds
import tensorflow as tf


class Dataset:
    """loading dataset for machine translation"""
    def __init__(self):
        """class constructor for dataset class"""
        self.data_train = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='train',
            as_supervised=True
        )
        self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='validation',
            as_supervised=True
        )
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )
        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

    def tokenize_dataset(self, data):
        """creates subword  tokenizers"""
        training_corpus_en = []
        training_corpus_pt = []
        for pt, en in data:
            training_corpus_en.append(en.numpy().decode('utf-8'))
            training_corpus_pt.append(pt.numpy().decode('utf-8'))

        pretrained_tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )
        pretrained_tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased'
        )
        vocab_size = 2**13
        self.tokenizer_pt = pretrained_tokenizer_pt.train_new_from_iterator(
            training_corpus_pt, vocab_size=vocab_size
        )
        self.tokenizer_en = pretrained_tokenizer_en.train_new_from_iterator(
            training_corpus_en, vocab_size=vocab_size
        )
        return self.tokenizer_pt, self.tokenizer_en

    def encode(self, pt, en):
        """Encodes a translation into tokens"""
        pt = pt.numpy().decode('utf-8')
        en = en.numpy().decode('utf-8')
        encoded_pt = self.tokenizer_pt(pt).input_ids
        encoded_en = self.tokenizer_en(en).input_ids
        vocab_size = 2**13
        start_token = vocab_size
        end_token = vocab_size + 1
        pt_tokens = [start_token] + encoded_pt[1:-1] + [end_token]
        en_tokens = [start_token] + encoded_en[1:-1] + [end_token]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """tensorflow wrapper for encode"""
        pt_tokens, en_tokens = tf.py_function(
            func=self.encode, inp=[pt, en], Tout=[tf.int64, tf.int64]
        )
        return pt_tokens, en_tokens
