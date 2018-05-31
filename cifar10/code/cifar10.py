# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the CIFAR-10 network.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use input() instead.
 inputs, labels = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
import os
import re
import sys
import tarfile

import cifar10_input
import tensorflow as tf
from six.moves import urllib

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
# 每批训练语料有128个样本
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
# 定义数据存放地址
tf.app.flags.DEFINE_string('data_dir', '../data',
                           """Path to the CIFAR-10 data directory.""")

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1  # Initial learning rate.

# If a model is trained with multiple GPU's prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.

    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    # tf.histogram_summary(tensor_name + '/activations', x)
    tf.summary.histogram(tensor_name + '/activations', x)
    # tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def distorted_inputs():
    """Construct distorted input for CIFAR training using the Reader ops.

    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.

    Raises:
      ValueError: If no data_dir
    """
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
    return cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=FLAGS.batch_size)


def inputs(eval_data):
    """Construct input for CIFAR evaluation using the Reader ops.

    Args:
      eval_data: bool, indicating if one should use the train or eval data set.

    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.

    Raises:
      ValueError: If no data_dir
    """
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
    return cifar10_input.inputs(eval_data=eval_data, data_dir=data_dir,
                                batch_size=FLAGS.batch_size)


def inference(images):
    """Build the CIFAR-10 model.

    Args:
      images: Images returned from distorted_inputs() or inputs().

    Returns:
      Logits.
    """
    # 所有参数都在cpu上
    with tf.device('/cpu:0'):
        # conv1
        # 随机产生卷积核
        # images：[128, 24, 24, 3] [batch, in_height, in_width, in_channels]
        # kernel：[5, 5, 3, 64]
        # 进行卷积操作，产生输出
        conv1 = tf.layers.conv2d(
                inputs=images,
                filters=64,
                strides=1,
                kernel_size=[5,5],
                padding='same',
                kernel_initializer=tf.truncated_normal_initializer(stddev=1e-4),
                use_bias=True,
                bias_initializer=tf.constant_initializer(0.0),
                activation=tf.nn.relu,
                name='conv1'
                )

        # 这里写入日志，用户tensor board输出
        _activation_summary(conv1)

        # pool1
        # 最大池化操作
        pool1 = tf.layers.max_pooling2d(
                inputs=conv1,
                pool_size=[3,3],
                strides=2,
                padding='same',
                name='pool1',
                )
        # norm1
        # 局部响应归一化
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

        conv2 = tf.layers.conv2d(
                inputs=norm1,
                filters=64,
                strides=1,
                kernel_size=[5,5],
                padding='same',
                kernel_initializer=tf.truncated_normal_initializer(stddev=1e-4),
                use_bias=True,
                bias_initializer=tf.constant_initializer(0.1),
                activation=tf.nn.relu,
                name='conv2',
                )

        _activation_summary(conv2)

        pool2 = tf.layers.max_pooling2d(
                inputs=conv2,
                pool_size=[3,3],
                strides=2,
                padding='same',
                name='pool2',
                )

        norm2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
        norm2_shape = norm2.get_shape()
        norm2_flat = tf.reshape(
                norm2,
                [norm2_shape[0], norm2_shape[1]*norm2_shape[2]*norm2_shape[3]],
                )

        local3 = tf.layers.dense(
                inputs=norm2_flat,
                units=384,
                activation=tf.nn.relu,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.04),
                bias_initializer=tf.constant_initializer(0.1),
                name='local3',
                )

        _activation_summary(local3)

        local4 = tf.layers.dense(
                inputs=local3,
                units=192,
                activation=tf.nn.relu,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.04),
                bias_initializer=tf.constant_initializer(0.1),
                name='local4',
                )

        _activation_summary(local3)

        local5 = tf.layers.dense(
                inputs=local4,
                units=NUM_CLASSES,
                kernel_initializer=tf.truncated_normal_initializer(stddev=1/192.0),
                bias_initializer=tf.constant_initializer(0.0),
                name='local5',
                )
        # print(conv1.get_shape())
        # print(pool1.get_shape())
        # print(norm1.get_shape())
        # print(conv2.get_shape())
        # print(pool2.get_shape())
        # print(norm2.get_shape())
        # print(local3.get_shape())
        # print(local4.get_shape())
        # print(softmax_linear.get_shape())
    return local5


def loss(logits, labels):
    """Add L2Loss to all the trainable variables.

    Add summary for for "Loss" and "Loss/avg".
    Args:
      logits: Logits from inference().
      labels: Labels from distorted_inputs or inputs(). 1-D tensor
              of shape [batch_size]

    Returns:
      Loss tensor of type float.
    """
    dense_labels = tf.one_hot(
            indices=labels,
            depth=NUM_CLASSES,
            )
    # Calculate the average cross entropy loss across the batch.
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=dense_labels,
        logits=logits,
        name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    """Add summaries for losses in CIFAR-10 model.

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        # tf.scalar_summary(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name + ' (raw)', l)
        # tf.scalar_summary(l.op.name, loss_averages.average(l))
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


def train(total_loss, global_step):
    """Train CIFAR-10 model.

    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.

    Args:
      total_loss: Total loss from loss().
      global_step: Integer Variable counting the number of training steps
        processed.
    Returns:
      train_op: op for training.
    """
    # Variables that affect learning rate.
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    # 学习率衰减
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    # tf.scalar_summary('learning_rate', lr)
    tf.summary.scalar('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    # 计算梯度，更新
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    # 在运行图之前，必须进行梯度计算
    with tf.control_dependencies([loss_averages_op]):
        # 初始化计算梯度的图，lr是学习率衰减
        opt = tf.train.GradientDescentOptimizer(lr)
        # total_loss计算梯度，返回的是一个list，包含更新的pair对(gradient, variable)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        # tf.histogram_summary(var.op.name, var)
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            # tf.histogram_summary(var.op.name + '/gradients', grad)
            tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op


def maybe_download_and_extract():
    """Download and extract the tarball from Alex's website."""
    dest_directory = FLAGS.data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                                                             float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath,
                                                 reporthook=_progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)
