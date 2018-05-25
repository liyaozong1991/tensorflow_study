# coding: utf8
import logging
logging.basicConfig(filename="./log_simple", level=logging.INFO, format="[%(levelname)s]\t%(asctime)s\tLINENO:%(lineno)d\t%(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logging.info('-'*50 + 'program start' + '-'*50)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# auto download data
logging.info('开始加载数据')
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
logging.info('加载数据完成')

import tensorflow as tf
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,W) + b)
y_ = tf.placeholder("float", [None,10])

cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
init = tf.initialize_all_variables()

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
config.gpu_options.allow_growth = False
sess = tf.Session(config=config)
sess.run(init)

logging.info('开始训练')
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    loss, _ = sess.run([cross_entropy, train_step], feed_dict={x: batch_xs, y_: batch_ys})
    logging.info('第{}轮训结束，损失：{}'.format(i, loss))

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
logging.info('模型在测试集上的准确率为：{:.2%}'.format(acc))
