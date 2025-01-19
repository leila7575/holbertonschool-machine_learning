#!/usr/bin/env python3
"""This module contains the function create_train_op."""


import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(
    X_train, Y_train, X_valid, Y_valid, layer_sizes, activations,
    alpha, iterations, save_path="/tmp/model.ckpt"
):
    """trains the neural network."""

    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])

    y_pred = forward_prop(x, layer_sizes, activations)

    loss = calculate_loss(y, y_pred)

    accuracy = calculate_accuracy(y, y_pred)

    train_op = create_train_op(loss, alpha)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        for i in range(iterations + 1):
            feed_dict = {x: X_train, y: Y_train}
            cost, acc = sess.run([loss, accuracy], feed_dict=feed_dict)

            feed_dict_valid = {x: X_valid, y: Y_valid}
            val_cost, val_acc = sess.run(
                [loss, accuracy], feed_dict=feed_dict_valid
            )

            if i == 0 or i % 100 == 0 or i == iterations:
                print(f"After {i} iterations:")
                print(f"\tTraining Cost: {cost}")
                print(f"\tTraining Accuracy: {acc}")
                print(f"\tValidation Cost: {val_cost}")
                print(f"\tValidation Accuracy: {val_acc}")

            sess.run(train_op, feed_dict=feed_dict)

        saver = tf.train.Saver()
        saver.save(sess, save_path)

    return save_path
