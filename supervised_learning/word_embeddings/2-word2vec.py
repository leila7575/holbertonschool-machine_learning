#!/usr/bin/env python3
"""word2vec_model for building and training word2vec_model"""


import gensim


def word2vec_model(
    sentences, vector_size=100, min_count=5, window=5,
    negative=5, cbow=True, epochs=5, seed=0, workers=1
):
    """Builds and trains word2vec model"""
    if cbow:
        sg = 0
    else:
        sg = 1
    model = gensim.models.Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
        seed=seed,
        negative=negative
    )
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count, epochs=epochs)
    return model
