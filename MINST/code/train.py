# coding: utf8
import logging
logging.basicConfig(filename="./log", level=logging.INFO, format="[%(levelname)s]\t%(asctime)s\tLINENO:%(lineno)d\t%(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logging.info('-'*20 + 'program start' + '-'*20)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# auto download data
logging.info('开始加载数据')
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
logging.info('加载数据完成')
import tensorflow as tf

# 定义数据和标签
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

# 数据格式进行转换，恢复结构信息
x_image = tf.reshape(x, [-1,28,28,1])

# 使用32个5*5的卷积核，激活函数使用relu
conv1 = tf.layers.conv2d(inputs=x_image,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        use_bias=True,
        )
# 池化层
pool1 = tf.layers.max_pooling2d(conv1, 2, 2)
# 第二个卷积层
conv2 = tf.layers.conv2d(inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        use_bias=True,
        )
# 第二个池化层
pool2 = tf.layers.max_pooling2d(conv2, 2, 2)
# 展开
pool_flat = tf.reshape(pool2, [-1, 7*7*64])
# 全连接
fc1 = tf.layers.dense(
        inputs=pool_flat,
        units=1024,
        activation=tf.nn.relu,
        )
# 丢弃
rate = tf.placeholder("float")
fc1_drop = tf.layers.dropout(
        inputs=fc1,
        rate=rate,
        )
# 全连接，softmax
y = tf.layers.dense(
        inputs=fc1_drop,
        units=10,
        activation=tf.nn.softmax,
        )

cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = False
sess = tf.Session(config=config)
sess.run(tf.initialize_all_variables())

for i in range(20000):
    batch = mnist.train.next_batch(50)
    _, accuracy_value = sess.run([train_step, accuracy], feed_dict={x: batch[0], y_: batch[1], rate: 0.5})
    if i % 100 == 0:
        logging.info("step {}, training accuracy {:.2%}".format(i, accuracy_value))

accuracy_value = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, rate: 1.0})
logging.info("test accuracy {:.2%}".format(accuracy_value))
