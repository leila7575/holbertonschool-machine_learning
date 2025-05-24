#!/usr/bin/env python3
"""Contains bag_of_words for bag of words embedding matrix"""


import numpy as np
import re
import string


def bag_of_words(sentences, vocab=None):
    """Creates bag of words embedding matrix"""
    words = []
    embeddings = []

    for sentence in sentences:
        sentence = re.sub(r"'s\b", "", sentence)
        preprocessed_sentence = sentence.translate(
            str.maketrans('', '', string.punctuation)
        )
        words += preprocessed_sentence.lower().split()
    if vocab is None:
        features = np.sort(list(set(words)))
    else:
        features = np.array(vocab)

    for sentence in sentences:
        embeddings_row = [0] * len(features)
        sentence = re.sub(r"'s\b", "", sentence)
        preprocessed_sentence = sentence.translate(
            str.maketrans('', '', string.punctuation)
        ).lower()
        for word in preprocessed_sentence.split():
            for i, val in np.ndenumerate(features):
                if word == val:
                    embeddings_row[i[0]] += 1
        embeddings.append(embeddings_row)

    return embeddings, features
