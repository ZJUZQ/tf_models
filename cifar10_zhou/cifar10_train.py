#!/usr/bin/env python

"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time

import tensorflow as tf

import cifar10

parser = cifar10.parser

parser.add_argument('--train_dir', type=str, default='/cifar10/cifar10_train',
                    help='Directory where to write event logs and checkpoint.')

parser.add_argument('--max_steps', type=int, default=10000,
                    help='Number of batches to run.')

parser.add_argument('--log_device_placement', type=bool, default=False,
                    help='Whether to log device placement.')

parser.add_argument('--log_frequency', type=int, default=10,
                    help='How often to log results to the console.')


def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.contrib.framework.get_or_create_global_step() # Returns and create (if necessary) the global step tensor 

    # Get images and labels for CIFAR-10.
    # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
    # GPU and resulting in a slow down.
    with tf.device('/cpu:0'):
      images, labels = cifar10.distorted_inputs() # Generate a batch of images and labels

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = cifar10.inference(images)

    # Calculate loss.
    loss = cifar10.loss(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = cifar10.train(loss, global_step)

    # class _LoggerHook inheritant tf.train.SessionRunHook( Hook to extend calls to MonitoredSession.run() )
    class _LoggerHook(tf.train.SessionRunHook): 
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):            # Called before each call to run().
        self._step += 1
        return tf.train.SessionRunArgs(loss)        # Asks for loss value.

      def after_run(self, run_context, run_values): # Called after each call to run().
        if self._step % FLAGS.log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          loss_value = run_values.results
          examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
          sec_per_batch = float(duration / FLAGS.log_frequency)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))

    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),  # Hook that requests stop at a specified step
               tf.train.NanTensorHook(loss),                        # Monitors the loss tensor and stops training if loss is NaN.
               _LoggerHook()],
        config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)) as mon_sess:
      while not mon_sess.should_stop():
        mon_sess.run(train_op)


def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.train_dir):          # Determines whether a path exists or not
    tf.gfile.DeleteRecursively(FLAGS.train_dir) # Deletes everything under dirname recursively
  tf.gfile.MakeDirs(FLAGS.train_dir)            # Creates a directory and all parent/intermediate directories
  train()


if __name__ == '__main__': # execute only if run as a script
  FLAGS = parser.parse_args()
  tf.app.run() # Runs the program with an optional 'main' function and 'argv' list
