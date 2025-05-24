#!/usr/bin/env python3
"""unigram Blue score computation"""


import numpy as np
from collections import Counter


def uni_bleu(references, sentence):
    """Computes unigram Blue score for a sentence"""
    clipping_val = {}
    for reference in references:
        reference_count = Counter(reference)
        for token in reference_count:
            clipping_val[token] = max(
                clipping_val.get(token, 0), reference_count[token]
            )

    count = 0
    sentence_count = Counter(sentence)
    for word in sentence:
        if clipping_val.get(word, 0) > 0:
            count += 1
            clipping_val[word] -= 1
    references_len = [len(reference) for reference in references]
    reference_len = min(
        references_len, key=lambda reference: abs(reference - len(sentence))
    )

    if len(sentence) > reference_len:
        bp = 1
    else:
        bp = np.exp(1 - (reference_len / len(sentence)))

    return bp * (count / len(sentence))
