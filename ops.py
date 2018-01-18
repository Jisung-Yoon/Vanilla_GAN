import tensorflow as tf


image_summary = tf.summary.image
scalar_summary = tf.summary.scalar
histogram_summary = tf.summary.histogram
merge_summary = tf.summary.merge
SummaryWriter = tf.summary.FileWriter


def linear_layer(inputs, output_size, scope=None, index='', stddev=0.01, with_vars=False):
    with tf.variable_scope(scope or "Linear"):

        weight_name = 'weight_' + str(index)
        bias_name = 'bias_' + str(index)

        weight = tf.get_variable(name=weight_name, shape=[inputs.get_shape()[1], output_size], dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable(name=bias_name, shape=[output_size],
                               initializer=tf.constant_initializer(0))
        layer_before_activation = tf.matmul(inputs, weight) + bias

    if with_vars:
        return layer_before_activation, weight, bias

    else:
        return layer_before_activation



