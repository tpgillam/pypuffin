''' Simple MNIST test using tensorflow. An example showing basic usage of tensorboard & convolutional layers. '''

import operator
import os

from functools import reduce

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


def weight_variable(shape):
    ''' A weight variable -- initialised with random small positive value '''
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    ''' A bias variable, initialised with constant non-zero value '''
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    ''' Apply the patch W over the input tensor x, which has dimensions

            (n_samples, height, width, channels)

        aka 'NHWC' format, which is the default.
    '''
    # Note that the first and last elements of strides *must* be one, according to the documentation
    stride = 1
    # padding is either SAME or VALID.
    #   VALID: no padding, so output image is smaller than input
    #   SAME: pads the input such that the output is the same size as the original input. Padding
    #         is, as far as possible, applied symmetrically. Padded values are filled with 0.
    return tf.nn.conv2d(x, W, strides=(1, stride, stride, 1), padding='SAME')


def max_pool(x, h, w):
    ''' Apply max pooling of a patch size (h, w) '''
    # We use SAME for the pooling, as we also used it for the convolution
    return tf.nn.max_pool(x, ksize=(1, h, w, 1), strides=(1, h, w, 1), padding='SAME')


def main():
    # TODO centralise dataset caching!
    data = input_data.read_data_sets('datasets/MNIST_data', one_hot=True)

    # The shape argument is a guide - an error will be thrown if we try
    # to assign a tensor of the wrong shape. None means that we don't place
    # a constraint on the extent of that dimension
    x = tf.placeholder(tf.float32, shape=(None, 784), name='x')
    y_ = tf.placeholder(tf.float32, shape=(None, 10), name='y_')

    # We convert the input to an image. They are monocrhomatic 28x28 images.
    # The -1 is a special placeholder value... it means to figure it out at graph evaluation time.
    n_channels_0 = 1
    x_image = tf.reshape(x, (-1, 28, 28, n_channels_0))

    # First convolutional layer
    with tf.name_scope('conv1'):
        n_channels_1 = 32
        W_conv_1 = weight_variable((5, 5, n_channels_0, n_channels_1))
        b_conv_1 = bias_variable((n_channels_1,))
        h_conv_1 = tf.nn.relu(conv2d(x_image, W_conv_1) + b_conv_1)
        h_pool_1 = max_pool(h_conv_1, 2, 2)

    # Second convolutional layer
    with tf.name_scope('conv2'):
        n_channels_2 = 64
        W_conv_2 = weight_variable((5, 5, n_channels_1, n_channels_2))
        b_conv_2 = bias_variable((n_channels_2,))
        h_conv_2 = tf.nn.relu(conv2d(h_pool_1, W_conv_2) + b_conv_2)
        h_pool_2 = max_pool(h_conv_2, 2, 2)

    # Third convolutional layer
    with tf.name_scope('conv3'):
        n_channels_3 = 64
        W_conv_3 = weight_variable((5, 5, n_channels_2, n_channels_3))
        b_conv_3 = bias_variable((n_channels_3,))
        h_conv_3 = tf.nn.relu(conv2d(h_pool_2, W_conv_3) + b_conv_3)
        h_pool_3 = max_pool(h_conv_3, 2, 2)


    with tf.name_scope('reshape'):
        # image_shape is of instance TensorShape. The elements are of type Dimension, so we get the
        # values, and compute the dimensionality of the flattened image
        image_shape = h_pool_3.get_shape()
        flat_dim = reduce(operator.mul, (dimension.value for dimension in image_shape[1:]))
        flattened = tf.reshape(h_pool_3, (-1, flat_dim))

    # Fully connected layer
    with tf.name_scope('fc1'):
        n_neurons_fcon_1 = 1024
        W_fcon_1 = weight_variable((flat_dim, n_neurons_fcon_1))
        b_fcon_1 = bias_variable((n_neurons_fcon_1,))
        h_fcon_1 = tf.nn.relu(tf.matmul(flattened, W_fcon_1) + b_fcon_1)

        # Dropout
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        h_dropout = tf.nn.dropout(h_fcon_1, keep_prob)

    # Fully connected layer
    with tf.name_scope('fc2'):
        n_neurons_fcon_2 = 1024
        W_fcon_2 = weight_variable((n_neurons_fcon_1, n_neurons_fcon_2))
        b_fcon_2 = bias_variable((n_neurons_fcon_2,))
        h_fcon_2 = tf.nn.relu(tf.matmul(h_dropout, W_fcon_2) + b_fcon_2)

        # Dropout
        h_dropout = tf.nn.dropout(h_fcon_2, keep_prob)

    with tf.name_scope('readout'):
        # Final readout layer
        W_fcon_3 = weight_variable((n_neurons_fcon_2, 10))
        b_fcon_3 = bias_variable((10,))

        # Form the prediction
        y = tf.matmul(h_dropout, W_fcon_3) + b_fcon_3


    with tf.name_scope('optimiser'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
        # train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(loss)
        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    # Model evaluation. We're taking an average in dimension 1, which implies a mean across
    # all samples in the current batch
    is_correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

    # Keep tabs on the accuracy
    # run_name = 'grad_desc_1em4'
    run_name = 'adam_1em4_4'
    tf.summary.scalar('accuracy', accuracy)
    all_summaries = tf.summary.merge_all()

    # Initialise variables last to take into account any that might have been created e.g. during
    # optimiser creation
    init_op = tf.global_variables_initializer()


    with tf.Session() as session:
        # TODO common path for putting tensorboard runs
        # Include graph information in the training writer
        train_writer = tf.summary.FileWriter(os.path.join(run_name, 'train'), session.graph)

        # Specify flush_secs to enforce more frequent dumping of things
        test_writer = tf.summary.FileWriter(os.path.join(run_name, 'test'), flush_secs=10)

        init_op.run()

        for i in range(20000):
            batch = data.train.next_batch(100)
            feed_train = {x: batch[0], y_: batch[1], keep_prob: 0.3}

            if i % 100 == 1:
                # Perform profiling
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                # Need to run via session so we can specify run options, and retrieve the metadata
                session.run(train_step, feed_dict=feed_train, options=run_options, run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, f'step_{i}', i)
            else:
                train_step.run(feed_dict=feed_train)

            # this is like 'evaluate_many' - we simulataneously evaluate the given nodes with
            # this particular feed_dict.
            summary, train_accuracy = session.run((all_summaries, accuracy),
                                                  feed_dict={**feed_train, keep_prob: 1})
            train_writer.add_summary(summary, i)


            if i % 100 == 0:
                feed_eval = {x: data.test.images, y_: data.test.labels, keep_prob: 1}
                summary, test_accuracy = session.run((all_summaries, accuracy), feed_dict=feed_eval)
                test_writer.add_summary(summary, i)
                print(f'{i}:  {train_accuracy}   {test_accuracy}')


if __name__ == '__main__':
    main()
