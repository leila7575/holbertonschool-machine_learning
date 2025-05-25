#!/usr/bin/env python3
"""Contains tf_idf for TF-IDF embedding"""


import numpy as np
import re
import string


def tf_idf(sentences, vocab=None):
    """Creates TF-IDF embedding"""

    preprocessed_sentences = []
    tokens = []
    for sentence in sentences:
        sentence = re.sub(r"'s\b", "", sentence)
        sentence = sentence.translate(
            str.maketrans('', '', string.punctuation)
        )
        sentence_tokens = sentence.lower().split()
        preprocessed_sentences.append(sentence_tokens)
        tokens += sentence_tokens

    if vocab is None:
        features = np.sort(list(set(tokens)))
    else:
        features = np.array(vocab)

    count_doc_with_word = [0] * len(features)
    for tokens in preprocessed_sentences:
        for i, val in np.ndenumerate(features):
            if val in tokens:
                count_doc_with_word[i[0]] += 1

    corpus_len = len(sentences)
    count_doc_with_word = np.array(count_doc_with_word)
    idf = np.log(corpus_len / (count_doc_with_word + 1))

    embeddings = []
    for tokens in preprocessed_sentences:
        tf = [0] * len(features)
        for word in tokens:
            for i, val in np.ndenumerate(features):
                if word == val:
                    tf[i[0]] += 1
        tf = np.array(tf) / len(tokens)
        tf_idf = tf * idf
        norm = np.linalg.norm(tf_idf)
        if norm > 0:
            tf_idf = tf_idf / norm
        embeddings.append(tf_idf)

    return np.array(embeddings), features
