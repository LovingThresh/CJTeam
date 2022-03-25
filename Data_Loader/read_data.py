# -*- coding: utf-8 -*-
# @Time    : 2022/3/24 22:15
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : read_data.py
# @Software: PyCharm

import cv2
import lmdb
import numpy as np

env = lmdb.open('lmdb_dir')

with env.begin(write=False) as txn:
    # 获取图像数据
    image_bin = txn.get('image_000'.encode())
    label = txn.get('label_000'.encode()).decode()  # 解码

    # 将二进制文件转为十进制文件（一维数组）
    image_buf = np.frombuffer(image_bin, dtype=np.uint8)
    # 将数据转换(解码)成图像格式
    # cv2.IMREAD_GRAYSCALE为灰度图，cv2.IMREAD_COLOR为彩色图
    img = cv2.imdecode(image_buf, cv2.IMREAD_COLOR)
    cv2.imshow('image', img)
    cv2.waitKey(0)
