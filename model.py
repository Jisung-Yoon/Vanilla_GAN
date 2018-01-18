'''
    Model: GAN
'''
import tensorflow as tf
from util import *
from ops import *


class GAN:
    def __init__(self,
                 sess,
                 latent_size=100,
                 input_size=28*28,
                 activation_function=tf.nn.relu,
                 optimizer=tf.train.AdamOptimizer,
                 learning_rate=0.0002,
                 name='GAN'):
        '''
        Args:
            latent_size: length of latent vector whichi is input of generator
            input_size: length of input vector (In MNIST case, input_size is 784)
            activation function or optimizer can be altered to other options.
        '''

        self.name = name
        self.result_path = os.path.join('./result', self.name)
        self.summary_path = './logs'
        check_and_make_dir()
        check_and_make_dir(self.result_path)
        self.counts = 0

        self.latent_size = latent_size
        self.input_size = input_size
        self.activation_function = activation_function

        # Make placeholder
        self.G_input = tf.placeholder(tf.float32, [None, self.latent_size], name='g_input')
        self.D_input = tf.placeholder(tf.float32, [None, self.input_size], name='d_input')

        # Make layer
        self.G = self.generator(self.G_input)
        self.D_logits_data, self.D_data = self.discriminator(self.D_input)
        self.D_logits_fake, self.D_fake = self.discriminator(self.G, reuse=True)

        # Define loss function
        self.D_loss_data = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_data, tf.ones_like(self.D_data)))
        self.D_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_fake, tf.zeros_like(self.D_fake)))
        self.D_loss = self.D_loss_data + self.D_loss_fake

        self.G_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_fake, tf.ones_like(self.D_fake)))

        # Set summary operator of loss functions
        self.D_summary_loss_data = scalar_summary("D_loss_data", self.D_loss_data)
        self.D_summary_loss_fake = scalar_summary("D_loss_fake", self.D_loss_fake)
        self.D_summary_loss = scalar_summary("D_loss", self.D_loss)
        self.G_summary_loss = scalar_summary("G_loss", self.G_loss)

        # Divide trainable variables and divide variables
        train_vars = tf.trainable_variables()
        self.D_vars = [x for x in train_vars if x.name.startswith('discriminator')]
        self.G_vars = [x for x in train_vars if x.name.startswith('generator')]

        # Define optimizer
        self.D_train = optimizer(learning_rate=learning_rate).minimize(self.D_loss, var_list=self.D_vars)
        self.G_train = optimizer(learning_rate=learning_rate).minimize(self.G_loss, var_list=self.G_vars)

        self.sess = sess
        self.sess.run(tf.global_variables_initializer())

        # Merge summaries and define summary writer
        self.G_summary = merge_summary([self.G_summary_loss])
        self.D_summary = merge_summary(
            [self.D_summary_loss, self.D_summary_loss_data, self.D_summary_loss_fake])
        self.writer = SummaryWriter(self.summary_path, self.sess.graph)

    # Train function (one_step)
    def train(self, g_input, d_input):
        D_feed_dict = {
            self.G_input: g_input,
            self.D_input: d_input
        }
        summary_str, d_loss, _ = self.sess.run([self.D_summary, self.D_loss, self.D_train], feed_dict=D_feed_dict)
        self.writer.add_summary(summary_str, self.counts)

        G_feed_dict = {
            self.G_input: g_input
        }
        summary_str, g_loss, _ = self.sess.run([self.G_summary, self.G_loss, self.G_train], feed_dict=G_feed_dict)
        self.writer.add_summary(summary_str, self.counts)

        self.counts += 1

        return d_loss, g_loss

    # Generator of GAN
    def generator(self, latent_var):
        with tf.variable_scope("generator") as scope:
            self.h1, self.W1, self.b1 = linear_layer(
                latent_var, 256, scope=scope, index=1, with_vars=True)
            self.h1 = self.activation_function(self.h1)

            self.h2, self.W2, self.b2 = linear_layer(
                self.h1, self.input_size, scope=scope, index=2, with_vars=True)
            self.h2 = tf.nn.sigmoid(self.h2)

        return self.h2

    # Discriminator of GAN
    def discriminator(self, inputs, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            h1 = linear_layer(inputs, 256, scope=scope, index=1)
            h1 = self.activation_function(h1)

            h2 = linear_layer(h1, 64, scope=scope, index=2)
            h2 = self.activation_function(h2)

            h3_logits = linear_layer(h2, 1, scope=scope, index=3)
            h3 = tf.nn.sigmoid(h3_logits)

        return h3_logits, h3

    # Generate images using given latent variables.
    def generating_images(self, g_input):
        feed_dict = {
            self.G_input: g_input
        }
        return self.sess.run(self.G, feed_dict=feed_dict)
