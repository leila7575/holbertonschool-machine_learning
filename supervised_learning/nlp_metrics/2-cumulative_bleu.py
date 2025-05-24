#!/usr/bin/env python3
"""cumulative n-gram Blue score computation"""


import numpy as np
from collections import Counter


def cumulative_bleu(references, sentence, n):
    """Computes cumulative n-gram Blue score for a sentence"""
    precisions = []
    for k in range(1, n + 1):
        ngrams = []
        for i in range(len(sentence) - k + 1):
            ngrams.append(" ".join(sentence[i: i + k]))

        references_count = []
        ngrams_ref = []
        clipping_val = {}
        for reference in references:
            for i in range(len(reference) - k + 1):
                ngrams_ref.append(" ".join(reference[i: i + k]))
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
        precisions.append(count / len(ngrams))

    combined_precision = np.exp(np.mean(np.log(precisions)))

    references_len = [len(reference) for reference in references]
    reference_len = min(
        references_len,
        key=lambda reference_len: abs(reference_len - len(sentence))
    )

    if len(sentence) > reference_len:
        bp = 1
    else:
        bp = np.exp(1 - (reference_len / len(sentence)))

    return bp * combined_precision
