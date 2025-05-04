#!/usr/bin/env python3
"""contains simple GAN model class"""


import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class Simple_GAN(keras.Model):
    """ Simple GAN model class"""
    def __init__(
        self, generator, discriminator, latent_generator,
        real_examples, batch_size=200, disc_iter=2, learning_rate=.005
    ):
        """Initializes GAn model class attributes."""
        super().__init__()    # run the __init__ of keras.Model first.
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = .5  # standard value, but can be changed if necessary
        self.beta_2 = .9  # standard value, but can be changed if necessary

        # define the generator loss and optimizer:
        self.generator.loss = lambda x: tf.keras.losses.MeanSquaredError()(
            x, tf.ones(x.shape)
        )
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1, beta_2=self.beta_2
        )
        self.generator.compile(
            optimizer=generator.optimizer, loss=generator.loss
        )

        # define the discriminator loss and optimizer:
        self.discriminator.loss = (
            lambda x, y: tf.keras.losses.MeanSquaredError()(
                x, tf.ones(x.shape)
            )
        ) + tf.keras.losses.MeanSquaredError()(y, -1*tf.ones(y.shape))
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1, beta_2=self.beta_2
        )
        self.discriminator.compile(
            optimizer=discriminator.optimizer, loss=discriminator.loss
        )

    # generator of real samples of size batch_size
    def get_fake_sample(self, size=None, training=False):
        """Generates a fake sample by applying
        the image of generator to latent sample"""

        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    # generator of fake samples of size batch_size
    def get_real_sample(self, size=None):
        """Generates a real sample, subset of real_examples set"""
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    # overloading train_step()
    def train_step(self, useless_argument):
        """Training loop for applying gradient descent to discriminator"""
        for _ in range(self.disc_iter):

            # compute the loss for the discriminator in a tape
            # watching the discriminator's weights
            with tf.GradientTape() as g_discr:
                # get a real sample
                real_sample = self.get_real_sample(self.batch_size)
                real_output = self.discriminator(real_sample)
                # get a fake sample
                fake_sample = self.get_fake_sample(self.batch_size)
                fake_output = self.discriminator(fake_sample)
                # compute the loss discr_loss of the discriminator
                # on real and fake samples
                discr_loss = self.discriminator.loss(real_output, fake_output)
            # apply gradient descent once to the discriminator
            gradients_discriminator = g_discr.gradient(
                discr_loss, self.discriminator.trainable_variables
            )
            self.discriminator.optimizer.apply_gradients(
                zip(
                    gradients_discriminator,
                    self.discriminator.trainable_variables
                )
            )

        # compute the loss for the generator in a tape
        # watching the generator's weights
        with tf.GradientTape() as g_gen:
            # get a fake sample
            generated_sample = self.get_fake_sample(self.batch_size)
            fake_sample = self.discriminator(generated_sample)
            # compute the loss gen_loss of the generator on this sample
            gen_loss = self.generator.loss(fake_sample)
        # apply gradient descent to the discriminator
        gradients_generator = g_gen.gradient(
            gen_loss, self.generator.trainable_variables
        )
        self.generator.optimizer.apply_gradients(
            zip(gradients_generator, self.generator.trainable_variables)
        )

        return {"discr_loss": discr_loss, "gen_loss": gen_loss}
