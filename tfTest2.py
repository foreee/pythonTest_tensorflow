# 进入mnist 初级训练
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./", one_hot=True)

x = tf.placeholder("float", [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 为了计算交叉熵，我们首先需要添加一个新的占位符用于输入正确值
y_ = tf.placeholder("float", [None, 10])
# 计算交叉熵
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
# 梯度下降的反向传播算法以0.01的学习速率最小化交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)
# 然后开始训练模型，这里我们让模型循环训练1000次！
# 该循环的每个步骤中，我们都会随机抓取训练数据中的100个批处理数据点，然后我们用这些数据点作为参数替换之前的占位符来运行train_step。
for i in range(5000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


# 评估我们的模型
# tf.argmax 能给出某个tensor对象在某一维上的其数据最大值所在的索引值，对比索引来判断结果是否正确
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# 把布尔值转换成浮点数，然后取平均值。
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# 最后计算所学习到的模型在测试数据集上面的正确率。
successPercent = sess.run(
    accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
print(successPercent)
print(sess.run(W)[2:20, 2], sess.run(b))

sess.close()
