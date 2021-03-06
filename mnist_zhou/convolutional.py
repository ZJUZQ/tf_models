#!/usr/bin/env python
#coding=utf-8

"""Simple, end-to-end, LeNet-5-like convolutional MNIST model example.

This should achieve a test error of 0.7%. Please keep this model as simple and
linear as possible, it is meant as a tutorial for simple convolutional models.
Run with --self_test on the command line to execute a short self-test.
"""
from __future__ import absolute_import
from __future__ import division # Python2.2以及以后的版本中增加了一个算术运算符" // "来表示整数除法，返回不大于结果的一个最大的整数
from __future__ import print_function

import argparse
import gzip
import os
import sys
import time
from datetime import datetime # use datetime.now()

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY = '/mnist'
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
VALIDATION_SIZE = 5000  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 64
NUM_EPOCHS = 10
EVAL_BATCH_SIZE = 64
EVAL_FREQUENCY = 100  # Number of steps between evaluations.


FLAGS = None


def data_type():
  """Return the type of the activations, weights, and placeholder variables."""
  if FLAGS.use_fp16:
    return tf.float16
  else:
    return tf.float32


def maybe_download(filename):
  """Download the data from Yann's website, unless it's already here."""
  if not tf.gfile.Exists(WORK_DIRECTORY):
    tf.gfile.MakeDirs(WORK_DIRECTORY)
  filepath = os.path.join(WORK_DIRECTORY, filename)
  if not tf.gfile.Exists(filepath):
    filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
    with tf.gfile.GFile(filepath) as f:
      size = f.size()
    print('Successfully downloaded', filename, size, 'bytes.')
  return filepath


def extract_data(filename, num_images):
  """Extract the images into a 4D tensor [image index, y, x, channels].

  Values are rescaled from [0, 255] down to [-0.5, 0.5].
  """
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(16) # first 16 bytes is the description informatoion of training set images
    buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
    data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
    data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH   # data normalization
    data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
    return data


def extract_labels(filename, num_images):
  """Extract the labels into a vector of int64 label IDs."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(8)  # first 8 bytes is the description informatoion of training set labels
    buf = bytestream.read(1 * num_images)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.int64)
  return labels


def fake_data(num_images):
  """Generate a fake dataset that matches the dimensions of MNIST."""
  data = numpy.ndarray(
      shape=(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS),
      dtype=numpy.float32)
  labels = numpy.zeros(shape=(num_images,), dtype=numpy.int64)
  for image in xrange(num_images):  # xrange 用法与 range 完全相同，所不同的是生成的不是一个list对象，而是一个生成器; 要生成很大的数字序列的时候，用xrange会比range性能优很多，因为不需要一上来就开辟一块很大的内存空间
    label = image % 2
    data[image, :, :, 0] = label - 0.5
    labels[image] = label
  return data, labels


def error_rate(predictions, labels):
  """Return the error rate based on dense predictions and sparse labels."""
  return 100.0 - (
      100.0 *
      numpy.sum(numpy.argmax(predictions, 1) == labels) /
      predictions.shape[0])


def main(_):
  if FLAGS.self_test:
    print('Running self-test.')
    train_data, train_labels = fake_data(256)
    validation_data, validation_labels = fake_data(EVAL_BATCH_SIZE)
    test_data, test_labels = fake_data(EVAL_BATCH_SIZE)
    num_epochs = 1
    """
      Because usually we divide the training set into batches, each epoch go through the 
      whole training set. Each iteration goes through on batch.

      You don’t just run through the training set once, it can take thousands of epochs for 
      your backpropagation algorithm to converge on a combination of weights with an acceptable 
      level of accuracy. Remember gradient descent only changes the weights by a small amount in 
      the direction of improvement, so backpropagation can’t get there by running through the 
      training examples just once.
    """
  else:
    # Get the data. 
    # The MNIST database of handwritten digits, available from this page, has a training set 
    # of 60,000 examples, and a test set of 10,000 examples
    train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
    train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
    test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
    test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')

    # Extract it into numpy arrays.
    train_data = extract_data(train_data_filename, 60000)
    train_labels = extract_labels(train_labels_filename, 60000)
    test_data = extract_data(test_data_filename, 10000)
    test_labels = extract_labels(test_labels_filename, 10000)

    # Generate a validation set.
    validation_data = train_data[:VALIDATION_SIZE, ...]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_data = train_data[VALIDATION_SIZE:, ...]
    train_labels = train_labels[VALIDATION_SIZE:]
    num_epochs = NUM_EPOCHS

  train_size = train_labels.shape[0]

  # This is where training samples and labels are fed to the graph.
  # These placeholder nodes will be fed a batch of training data at each
  # training step using the {feed_dict} argument to the Run() call below.
  with tf.name_scope('train_data_node'):
    train_data_node = tf.placeholder(
        data_type(),
        shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
  with tf.name_scope('train_labels_node'):
    train_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))
  with tf.name_scope('eval_data'):
    eval_data = tf.placeholder(
        data_type(),
        shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

  # The variables below hold all the trainable weights. They are passed an
  # initial value which will be assigned when we call:
  # {tf.global_variables_initializer().run()}
  with tf.name_scope('conv1_weights'):
    conv1_weights = tf.Variable(
        tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                            stddev=0.1,
                            seed=SEED, dtype=data_type()))
  with tf.name_scope('conv1_biases'):
    conv1_biases = tf.Variable(tf.zeros([32], dtype=data_type()))

  with tf.name_scope('conv2_weights'):
    conv2_weights = tf.Variable(tf.truncated_normal(
        [5, 5, 32, 64], stddev=0.1,
        seed=SEED, dtype=data_type()))
  with tf.name_scope('conv2_biases'):
    conv2_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=data_type()))

  with tf.name_scope('fc1_weights'):
    fc1_weights = tf.Variable(  # fully connected, depth 512.
        tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512],
                            stddev=0.1,
                            seed=SEED,
                            dtype=data_type()))
  with tf.name_scope('fc1_biases'):
    fc1_biases = tf.Variable(tf.constant(0.1, shape=[512], dtype=data_type()))

  with tf.name_scope('fc2_weights'):
    fc2_weights = tf.Variable(tf.truncated_normal([512, NUM_LABELS],
                                                  stddev=0.1,
                                                  seed=SEED,
                                                  dtype=data_type()))
  with tf.name_scope('fc2_biases'):
    fc2_biases = tf.Variable(tf.constant(
        0.1, shape=[NUM_LABELS], dtype=data_type()))

  # We will replicate the model structure for the training subgraph, as well
  # as the evaluation subgraphs, while sharing the trainable parameters.
  def model(data, train=False):
    """The Model definition."""
    # 2D convolution, with 'SAME' padding (i.e. the output feature map has
    # the same size as the input). Note that {strides} is a 4D array whose
    # shape matches the data layout: [image index, y, x, depth].
    with tf.name_scope('conv1'):
      conv1 = tf.nn.conv2d(data,
                          conv1_weights,
                          strides=[1, 1, 1, 1],
                          padding='SAME')
      # Bias and rectified linear non-linearity.
      relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    # Max pooling. The kernel size spec {ksize} also follows the layout of
    # the data. Here we have a pooling window of 2, and a stride of 2.
    with tf.name_scope('pool1'):
      pool1 = tf.nn.max_pool(relu1,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME')
    
    with tf.name_scope('conv2'):
      conv2 = tf.nn.conv2d(pool1,
                          conv2_weights,
                          strides=[1, 1, 1, 1],
                          padding='SAME')
      relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
    with tf.name_scope('pool2'):
      pool2 = tf.nn.max_pool(relu2,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME')
    # Reshape the feature map cuboid into a 2D matrix to feed it to the
    # fully connected layers.
    pool2_shape = pool2.get_shape().as_list()  # [N, H, W, C]
    with tf.name_scope('pool2_reshape'):
      pool2_reshape = tf.reshape(
          pool2,
          [pool2_shape[0], pool2_shape[1] * pool2_shape[2] * pool2_shape[3]])

    # Fully connected layer. Note that the '+' operation automatically
    # broadcasts the biases.
    with tf.name_scope('fc1'):
      fc1 = tf.nn.relu(tf.matmul(pool2_reshape, fc1_weights) + fc1_biases)

    # Add a 50% dropout during training only. Dropout also scales
    # activations such that no rescaling is needed at evaluation time.
    if train:
      fc1_dropout = tf.nn.dropout(fc1, 0.5, seed=SEED)
      return tf.matmul(fc1_dropout, fc2_weights) + fc2_biases

    # when test
    return tf.matmul(fc1, fc2_weights) + fc2_biases

  # Training computation: logits + cross-entropy loss.
  with tf.name_scope('logits'):
    logits = model(train_data_node, True)

  with tf.name_scope('cross'):
    with tf.name_scope('cross_entropy'):
      loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=train_labels_node, logits=logits))

    # L2 regularization for the fully connected parameters.
    with tf.name_scope('regularizers'):
      regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                      tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
    # Add the regularization term to the loss.
    loss += 5e-4 * regularizers

  # Optimizer: set up a variable that's incremented once per batch and
  # controls the learning rate decay.
  with tf.name_scope('batch_number'):
    batch = tf.Variable(0, dtype=data_type())
  
  # Decay once per epoch, using an exponential schedule starting at 0.01.
  with tf.name_scope('learning_rate'):
    learning_rate = tf.train.exponential_decay(   # decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
        0.01,                # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        train_size,          # Decay step.
        0.95,                # Decay rate.
        staircase=True)

  # Use simple momentum for the optimization.
  with tf.name_scope('optimizer'):
    optimizer = tf.train.MomentumOptimizer(learning_rate,
                                           0.9).minimize(loss,
                                                         global_step=batch)

  # Predictions for the current training minibatch.
  with tf.name_scope('train_prediction'):
    train_prediction = tf.nn.softmax(logits)

  # Predictions for the test and validation, which we'll compute less often.
  with tf.name_scope('eval_prediction'):
    eval_prediction = tf.nn.softmax(model(eval_data))

  # Small utility function to evaluate a dataset by feeding batches of data to
  # {eval_data} and pulling the results from {eval_predictions}.
  # Saves memory and enables this to run on smaller GPUs.
  def eval_in_batches(data, sess):
    """Get all predictions for a dataset by running it in small batches."""
    size = data.shape[0]
    if size < EVAL_BATCH_SIZE:
      raise ValueError("batch size for evals larger than dataset: %d" % size)
    predictions = numpy.ndarray(shape=(size, NUM_LABELS), dtype=numpy.float32)
    for begin in xrange(0, size, EVAL_BATCH_SIZE):
      end = begin + EVAL_BATCH_SIZE
      if end <= size:
        predictions[begin:end, :] = sess.run(
            eval_prediction,
            feed_dict={eval_data: data[begin:end, ...]})
      else:
        batch_predictions = sess.run(
            eval_prediction,
            feed_dict={eval_data: data[-EVAL_BATCH_SIZE:, ...]})
        predictions[begin:, :] = batch_predictions[begin - size:, :]
    return predictions

  logs_dir = './logs/train'
  if tf.gfile.Exists(logs_dir):
    tf.gfile.DeleteRecursively(logs_dir)
  tf.gfile.MakeDirs(logs_dir)
  
  # Create a local session to run the training.
  start_time = time.time()
  with tf.Session() as sess:
    # sess.graph contains the graph definition; that enables the Graph Visualizer
    train_writer = tf.summary.FileWriter('./logs/train', sess.graph)  # TensorBoard requires a logdir to read logs from

    # Run all the initializers to prepare the trainable parameters.
    tf.global_variables_initializer().run()
    print('Initialized!')
    # Loop through training steps.
    for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):
      # Compute the offset of the current minibatch in the data.
      # Note that we could use better randomization across epochs.
      offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
      batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
      batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
      # This dictionary maps the batch data (as a numpy array) to the
      # node in the graph it should be fed to.
      feed_dict = {train_data_node: batch_data,
                   train_labels_node: batch_labels}
      # Run the optimizer to update weights.
      sess.run(optimizer, feed_dict=feed_dict)
      #train_writer.add_summary(summary, step)

      # print some extra information once reach the evaluation frequency
      if step % EVAL_FREQUENCY == 0:
        # fetch some extra nodes' data
        l, lr, predictions = sess.run([loss, learning_rate, train_prediction],
                                      feed_dict=feed_dict)
        elapsed_time = time.time() - start_time   # Return the time in seconds
        start_time = time.time()
        print (datetime.now())
        print('Step %d (epoch %.2f), %.1f ms/batch' 
              %(step, 
                float(step) * BATCH_SIZE / train_size,
                1000 * elapsed_time / EVAL_FREQUENCY))
        print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
        print('Minibatch error: %.1f%%' % error_rate(predictions, batch_labels))  # %% will output a single %
        print('Validation error: %.1f%%' % error_rate(
            eval_in_batches(validation_data, sess), validation_labels))
        sys.stdout.flush()
        """
          Python's standard out is buffered (meaning that it collects some of the data "written" 
          to standard out before it writes it to the terminal). Calling sys.stdout.flush() forces 
          it to "flush" the buffer, meaning that it will write everything in the buffer to the 
          terminal, even if normally it would wait before doing so.
        """
      
    # Finally print the test result!
    test_error = error_rate(eval_in_batches(test_data, sess), test_labels)
    print('Test error: %.1f%%' % test_error)
    if FLAGS.self_test:
      print('test_error', test_error)
      assert test_error == 0.0, 'expected 0.0 test_error, got %.2f' % (
          test_error,)

    train_writer.close()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--use_fp16',
      default=False,
      help='Use half floats instead of full floats if True.',
      action='store_true')
  parser.add_argument(
      '--self_test',
      default=False,
      action='store_true',
      help='True if running a self test.')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)  # Runs the program with an optional 'main' function and 'argv' list
