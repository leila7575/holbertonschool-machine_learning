#!/usr/bin/env python3
"""This module contains the function create_train_op."""


import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def evaluate(X, Y, save_path):
    """Evaluates the output of neural network."""
    x, y = create_placeholders(X.shape[1], Y.shape[1])
    y_pred = forward_prop(X)
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, save_path)
        feed_dict = {x: X, y: Y}
        cost, acc, pred = sess.run([loss, accuracy, y_pred], feed_dict=feed_dict)

    return pred, acc, cost

