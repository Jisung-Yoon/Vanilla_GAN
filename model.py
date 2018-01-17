import tensorflow as tf
from util import *
from ops import *

class GAN:
    def __init__(self,
                 sess,
                 latent_size=100,
                 input_size=28 * 28,
                 activation_function=tf.nn.relu,
                 optimizer=tf.train.AdamOptimizer,
                 learning_rate=0.0002):

        self.latent_size = latent_size
        self.input_size = input_size
        self.activation_function = activation_function

        self.G_input = tf.placeholder(tf.float32, [None, self.latent_size], name='g_input')
        self.D_input = tf.placeholder(tf.float32, [None, self.input_size], name='d_input')

        # Define loss function
        self.G = self.generator(self.G_input)
        self.G_data = self.discriminator(self.D_input)
        self.G_fake = self.discriminator(self.G, reuse=True)
        self.D_loss = - (tf.reduce_mean(self.G_data) + tf.log(1 - self.G_fake))
        self.G_loss = - tf.reduce_mean(tf.log(self.G_fake))

        train_vars = tf.trainable_variables()
        self.D_vars = [x for x in train_vars if x.name.startswith('generator')]
        self.G_vars = [x for x in train_vars if x.name.startswith('discriminator')]

        self.D_train = optimizer(learning_rate=learning_rate).minimize(self.D_loss, var_list=self.D_vars)
        self.G_train = optimizer(learning_rate=learning_rate).minimize(self.G_loss, var_list=self.G_vars)

        self.sess = sess
        self.sess.run(tf.global_variables_initializer())

    # Train function
    def train(self, g_input, d_input):
        feed_dict_D = {
            self.G_input: g_input,
            self.D_input: d_input
        }
        d_loss, _ = self.sess.run([self.loss_D, self.train_D], feed_dict=feed_dict_D)

        feed_dict_G = {
            self.G_input: g_input
        }
        g_loss, _ = self.sess.run([self.loss_G, self.train_G], feed_dict=feed_dict_G)

        return d_loss, g_loss

    def generator(self, latent_var):
        with tf.variable_scope("generator") as scope:
            self.h1, self.W1, self.b1 = linear_layer(
                latent_var, 256, scope=scope, index=1, with_vars=True)
            self.h1 = self.activation_function(self.h1)

            self.h2, self.W2, self.b2 = linear_layer(
                latent_var, self.input_size, scope=scope, index=2, with_vars=True)
            self.h2 = tf.nn.sigmoid(self.h2)

        return self.h2

    def discriminator(self, inputs, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            h1 = linear_layer(inputs, 256, scope=scope, index=1)
            h1 = self.activation_function(h1)

            h2 = linear_layer(h1, 64, scope=scope, index=2)
            h2 = self.activation_function(h2)

            h3 = linear_layer(h2, 1, scope=scope, index=3)
            h3 = tf.nn.sigmoid(h3)

        return h3

    def generating_images(self, g_input):
        feed_dict = {
            self.G_input: g_input
        }
        return self.sess.run(self.G, feed_dict=feed_dict)
