import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)


def weights(shape):
    return tf.Variable(tf.truncated_normal(shape))
eewqe=7

def simple_layer(inputs, input_size, output_size):
    """
    封装一层神经网络
    :param inputs: 数据
    :param input_size: 行数
    :param output_size: 列数
    :return: 无
    """
    w = tf.Variable(tf.truncated_normal([input_size, output_size]))
    # w = tf.Variable(tf.random_normal([input_size,output_size]))
    b = tf.Variable(tf.zeros([output_size]) + 0.1)
    return tf.matmul(inputs, w) + b


def conv2d_layer(inputs, filter, strides, padding, activation=None):
    output = tf.nn.conv2d(inputs, filter, strides, padding)
    return activation(output)


def max_pooling(inputs, kernel_size, strides, padding):
    return tf.nn.max_pool(inputs, kernel_size, strides, padding)


def draw_graph(data_array, x_label, y_label, name, order):
    """
    绘制loss图像
    :param data_array: 数据
    :param x_label: x轴名称
    :param y_label: y轴名称
    :param name: 图表名称
    :return: 无
    """
    plt.figure(order)
    plt.plot(data_array)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title("lr={0} tr={1} bs={2}".format(learning_rate, train_epochs, batch_size))
    plt.savefig(name, dpi=200)


# 参数设置
learning_rate = 0.001  # 学习率
train_epochs = 25  # 训练轮数
batch_size = 100  # 一批数据的大小为100
display_step = 1  # 展示的步长
loss_record = []  # 储存loss
accuracy_record = []  # 存储accuracy

x = tf.placeholder(tf.float32, [None, 784])  # mnist 维度28*28 = 784
y = tf.placeholder(tf.float32, [None, 10])  # 表示0-9个数字的

x_image = tf.reshape(x, [-1, 28, 28, 1])  # 将[None,784]转换为[None,28,28,1]的形状
# 卷积层
conv2d_layer1 = conv2d_layer(x_image, weights([5, 5, 1, 16]), [1, 1, 1, 1], "SAME", tf.nn.relu)
conv2d_pooling1 = max_pooling(conv2d_layer1, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")  # [14,14,16]
# 卷积层
conv2d_layer2 = conv2d_layer(conv2d_pooling1, weights([5, 5, 16, 32]), [1, 1, 1, 1], "SAME", tf.nn.relu)
conv2d_pooling2 = max_pooling(conv2d_layer2, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")  # [7,7,32]

reshape = tf.reshape(conv2d_pooling2, [-1, 7 * 7 * 32])  # 展平[7*7*32]
# 全连接层
pred = tf.nn.softmax(simple_layer(reshape, 7 * 7 * 32, 1024))

# 反向传播，将生成的pred与y进行一次交叉熵运算最小化误差cost
# loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))   #reduce_sum是按列求和
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
# 梯度下降优化器
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
init = tf.global_variables_initializer()

saver = tf.train.Saver()  # 获取保存对象
model_path = "CNN3/model.cpkt"  # 设置保存模型的位置

# 启动session，训练模型
with tf.Session() as sess:
    sess.run(init)  # 初始化全部的值

    # 启动循环开始训练
    for epoch in range(train_epochs):
        train_loss_epoch = 0
        train_accuracy_epoch = 0

        # 将图片分组
        total_batch = int(mnist.train.num_examples / batch_size)

        # 遍历全部数据集
        for i in range(total_batch):
            # 获取一个批次的数据
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)

            # 启动optimizer优化器
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            # pred_s = sess.run(pred,feed_dict={x:batch_xs,y:batch_ys})
            # print(pred_s)
            train_loss = sess.run(loss, feed_dict={x: batch_xs, y: batch_ys})
            train_correct = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys})

            # 计算一次epochs的误差与准确率
            train_loss_epoch += train_loss / total_batch
            train_accuracy_epoch += train_correct / total_batch

        # 显示训练中的详细信息，50步展示一次
        if (epoch + 1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), " lost=", '{:.6f}'.format(train_loss_epoch), " accuracy=",
                  "{:.6f}".format(train_accuracy_epoch))
        loss_record.append(train_loss_epoch)
        accuracy_record.append(train_accuracy_epoch)

    print("Training finished!")

    # 存储模型
    save_path = saver.save(sess, model_path)
    print("Model saved in file:%s" % save_path)
    # 绘制图表
    draw_graph(loss_record, "train_epochs", "loss", "CNN_loss_graph3.jpg", 1)
    draw_graph(accuracy_record, "train_epochs", "accuracy", "CNN_accuracy_graph3.jpg", 2)
    print("Graph saved")
