# -*- coding: utf-8 -*-
# @Time    : 2022/2/19 7:59
# @Author  : Liuyee
# @Email   : SY2113205@buaa.edu.cn / csu1704liuye@163.com
# @File    : data_loader.py
# @Software: PyCharm

# 构建TF的data_loader

import tensorflow as tf
import numpy as np


def transform(image):
    image = image / 255
    return image


def load_function(img_path, Transform, Dict, target_size=(32, 32)):
    # 从路径导入图像，这一步可以使用opncv, tensorflow, PIL等库，可以根据自己要进行的图像处理库进行选择

    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image)
    if target_size != (0, 0):
        image = tf.image.resize(image, target_size)
    if Transform:
        image = Transform(image)
    # 如果是有监督，对应的image还需要Label
    # 在此案例中，Image分为了Apple与Orange,所以我们可以构建一个字典来进行标注
    label = tf.constant(2, dtype=tf.int8)
    if Dict:
        for key in Dict.keys():
            if tf.strings.regex_full_match(img_path, ".*{}.*".format(key)):
                label = tf.constant(Dict.get(key), dtype=tf.int8)

    return image, label


file_txt = r'L:\ALASegmentationNets_v2\Data\Stage_4\train.txt'
files_txt = np.loadtxt(file_txt, delimiter=',', dtype=bytes, encoding='utf-8')
path_ds = tf.data.Dataset.from_tensor_slices(files_txt)


def printt(ima):
    print(str(ima[0]))
    return tf.constant(1), ima


a = path_ds.map(printt)
for i, (images, ima) in enumerate(a.take(4)):
    print(images.shape)
