#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/2/4
# @Author  : Liu Ye
# @File    : .py
# @IDE: PyCharm

import tensorflow as tf
import numpy as np


# 将模型转换成TensorFlow Lite格式
def Converter(model):
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir=model)
    tflite_model = converter.convert()

    # save the model to the disk
    open("model.tflite", "wb").write(tflite_model)


# 将模型转换成量化后的TensorFlow Lite格式
def OptimizationConverter(model, x_test):
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir=model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset_generator():
        for value in x_test:
            yield [np.array(value, dtype=np.float32, ndim=2)]

    converter.representative_dataset = representative_dataset_generator()
    tflite_model = converter.convert()

    open("optimizationModel.tflite", "wb").write(tflite_model)

