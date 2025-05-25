#!/usr/bin/env python3
"""Using a trained Word2Vec gensim model to a keras embedding layer."""


from tensorflow import keras as K


def gensim_to_keras(model):
    """converts a gensim word2vec model to a keras Embedding layer."""
    word_vectors = model.wv
    word_embeddings = word_vectors.vectors
    embedding_layer = K.layers.Embedding(
        input_dim=word_embeddings.shape[0],
        output_dim=word_embeddings.shape[1],
        weights=[word_embeddings]
    )
    return embedding_layer
