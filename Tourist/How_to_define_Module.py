# -*- coding: utf-8 -*-
# @Time    : 2022/4/3 20:51
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : How_to_define_Module.py
# @Software: PyCharm

import datetime
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt


# 根据官网上的范例进行SimpleModule
class SimpleModule(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.a_variable = tf.Variable(5.0, name='train_me')
        self.non_trainable_variable = tf.Variable(5.0, trainable=False, name="do_not_train_me")

    def __call__(self, x, *args, **kwargs):
        return self.a_variable * x + self.non_trainable_variables


simple_module = SimpleModule(name='simple')

simple_module(tf.constant(5.0))


class FlexibleDenseModule(tf.Module):
    # Note: No need for `in+features`
    def __init__(self, out_features, name=None):
        super().__init__(name=name)
        self.is_built = False
        self.out_features = out_features

    def __call__(self, x):
        # Create variables on first call.
        if not self.is_built:
            self.w = tf.Variable(
                tf.random.normal([x.shape[-1], self.out_features]), name='w')
            self.b = tf.Variable(tf.zeros([self.out_features]), name='b')
            self.is_built = True

        y = tf.matmul(x, self.w) + self.b
        return tf.nn.relu(y)


# Used in a module
class MySequentialModule(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)

        self.dense_1 = FlexibleDenseModule(out_features=3)
        self.dense_2 = FlexibleDenseModule(out_features=2)

    def __call__(self, x):
        x = self.dense_1(x)
        return self.dense_2(x)


my_model = MySequentialModule(name="the_model")
print("Model results:", my_model(tf.constant([[2.0, 2.0, 2.0]])))

# CheckPoint
checkpoint_path = "my_checkpoint"
checkpoint = tf.train.Checkpoint(model=my_model)
checkpoint.write(checkpoint_path)

new_model = MySequentialModule()
new_checkpoint = tf.train.Checkpoint(model=new_model)
new_checkpoint.restore("my_checkpoint")

# Should be the same result as above
new_model(tf.constant([[2.0, 2.0, 2.0]]))


class Dense(tf.Module):
    def __init__(self, in_features, out_features, name=None):
        super().__init__(name=name)
        self.w = tf.Variable(
            tf.random.normal([in_features, out_features]), name='w')
        self.b = tf.Variable(tf.zeros([out_features]), name='b')

    def __call__(self, x):
        y = tf.matmul(x, self.w) + self.b
        return tf.nn.relu(y)


class MySequentialModule(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)

        self.dense_1 = Dense(in_features=3, out_features=3)
        self.dense_2 = Dense(in_features=3, out_features=2)

    @tf.function
    def __call__(self, x):
        x = self.dense_1(x)
        return self.dense_2(x)


# You have made a model with a graph!
my_model = MySequentialModule(name="the_model")

# Set up logging.
stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = "logs/func/%s" % stamp
writer = tf.summary.create_file_writer(logdir)

# Create a new model to get a fresh trace
# Otherwise the summary will not see the graph.
new_model = MySequentialModule()

# Bracket the function call with
# tf.summary.trace_on() and tf.summary.trace_export().
tf.summary.trace_on(graph=True, profiler=True)
# Call only one tf.function when tracing.
print(new_model(tf.constant([[2.0, 2.0, 2.0]])))
with writer.as_default():
    tf.summary.trace_export(
        name="my_func_trace",
        step=0,
        profiler_outdir=logdir)

tf.saved_model.save(my_model, 'the_saved_model')


# 定义模型
# 如果是使用tf.Module定义的，调用的方法是__call__
# 如果是使用tf.keras.layer.Layer定义层，调用的方法是call
class Mymodel(tf.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = tf.Variable(5.0)
        self.b = tf.Variable(0.0)

    def __call__(self, x, *args, **kwargs):
        return self.w * x + self.b


my_model = Mymodel()

TRUE_W = 3.0
TRUE_B = 2.0

NUM_EXAMPLES = 1000

# 随机向量x
x = tf.random.normal(shape=[NUM_EXAMPLES])

# 生成噪声
noise = tf.random.normal(shape=[NUM_EXAMPLES])

# 计算y
y = x * TRUE_W + TRUE_B + noise


# 定义损失函数
def loss(target_y, predicted_y):
    return tf.reduce_mean(tf.square(target_y - predicted_y))


# 定义训练循环
# 训练循环按顺序重读执行以下任务
# 1、发送一批输入值，通过模型生成输出值
# 2、通过比较输入值与输出（标签），来计算损失值
# 3、通过梯度带（GradientTape）找到梯度
# 4、通过这些梯度优化变量

# 给定一个可调用的模型，输入，输出和学习率

def train(model, x, y, learning_rate):
    with tf.GradientTape() as t:
        current_loss = loss(y, model(x))

    dw, db = t.gradient(current_loss, [model.w, model.b])

    model.w.assign_sub(learning_rate * dw)
    model.b.assign_sub(learning_rate * db)


Ws, bs = [], []
epochs = range(20)


# 定义用于训练的循环
def training_loop(model, x, y):
    for epoch in epochs:
        # 用单个大批量处理更新模型
        train(model, x, y, learning_rate=0.1)

        # 在更新之前进行更新
        Ws.append(model.w.numpy())
        bs.append(model.b.numpy())
        current_loss = loss(y, model(x))

        print("Epoch %2d: W=%1.2f b=%1.2f, loss=%2.5f" % (epoch, Ws[-1], bs[-1], current_loss))


print("Starting:W=%1.2f b=%1.2f, loss=%2.5f" % (my_model.w, my_model.b, loss(y, my_model(x))))

# 开始训练
training_loop(my_model, x, y)

# 绘制
plt.plot(epochs, Ws, "r",
         epochs, bs, "b")

plt.plot([TRUE_W] * len(epochs), "r--",
         [TRUE_B] * len(epochs), "b--")

plt.legend(["W", "b", "True W", "True b"])
plt.show()

# 可视化训练后的模型如何执行
plt.scatter(x, y, c="b")
plt.scatter(x, my_model(x), c="r")
plt.show()

print("Current loss: %1.6f" % loss(my_model(x), y).numpy())
