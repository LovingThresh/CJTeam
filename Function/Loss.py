#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/12/16
# @Author  : Liu Ye
# @File    : .py
# @IDE: PyCharm
import tensorflow as tf
import tensorflow.keras.backend as K
# 构建Focal loss用于数据不平衡
# Focal loss主要用于处理正负样本比例严重失调的问题


def Binary_Focal_loss(gamma=2, alpha=0.25):
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def binary_focal_loss(y_true, y_pred):
        y_true = y_true[:, :, :, 0]
        y_pred = y_pred[:, :, :, 0]

        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)

        p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred) + K.epsilon()
        focal_loss = -alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)

        return K.mean(focal_loss)

    return binary_focal_loss
