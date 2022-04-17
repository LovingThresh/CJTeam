# -*- coding: utf-8 -*-
# @Time    : 2022/4/3 20:51
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : How_to_define_Module.py
# @Software: PyCharm

import datetime
import tensorflow as tf
from tensorflow.keras import layers


class SimpleModule(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.a_variable = tf.Variable(5.0, name='train_me')
        self.non_trainable_variable = tf.Variable(5.0, trainable=False, name="do_not_train_me")

    def __call__(self, x, *args, **kwargs):
        return self.a_variable * x + self.non_trainable_variables


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


class MySequentialModule(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)

        self.dense_1 = FlexibleDenseModule(out_features=3)
        self.dense_2 = FlexibleDenseModule(out_features=2)

    def __call__(self, x):
        x = self.dense_1(x)
        return self.dense_2(x)


class Dense(tf.Module):
    def __init__(self, in_features, out_features, name=None):
        super().__init__(name=name)
        self.w = tf.Variable(
            tf.random.normal([in_features, out_features]), name='w')
        self.b = tf.Variable(tf.zeros([out_features]), name='b')

    def __call__(self, x):
        y = tf.matmul(x, self.w) + self.b
        return tf.nn.relu(y)


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


def train(model, x, y, learning_rate):
    with tf.GradientTape() as t:
        current_loss = loss(y, model(x))

    dw, db = t.gradient(current_loss, [model.w, model.b])

    model.w.assign_sub(learning_rate * dw)
    model.b.assign_sub(learning_rate * db)


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

