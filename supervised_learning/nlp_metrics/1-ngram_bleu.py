#!/usr/bin/env python3
"""n-gram Blue score computation"""


import numpy as np
from collections import Counter


def ngram_bleu(references, sentence, n):
    """Computes n-gram Blue score for a sentence"""
    ngrams = []
    for i in range(len(sentence) - n + 1):
        ngrams.append(" ".join(sentence[i: i + n]))

    references_count = []
    ngrams_ref = []
    clipping_val = {}
    for reference in references:
        for i in range(len(reference) - n + 1):
            ngrams_ref.append(" ".join(reference[i: i + n]))
        reference_count = Counter(ngrams_ref)
        references_count.append(reference_count)

    for ref_count in references_count:
        for ngram in ref_count:
            clipping_val[ngram] = max(
                clipping_val.get(ngram, 0), ref_count[ngram]
            )

    count = 0
    sentence_count = Counter(ngrams)
    for ngram in sentence_count:
        count += min(sentence_count[ngram], clipping_val.get(ngram, 0))
    references_len = [len(reference) for reference in references]
    reference_len = min(
        references_len,
        key=lambda reference_len: abs(reference_len - len(sentence))
    )

    if len(sentence) > reference_len:
        bp = 1
    else:
        bp = np.exp(1 - (reference_len / len(sentence)))

    return bp * (count / len(ngrams))
