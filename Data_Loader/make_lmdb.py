# -*- coding: utf-8 -*-
# @Time    : 2022/3/24 22:15
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : make_lmdb.py
# @Software: PyCharm

import lmdb


# 单张图片的lmdb数据集制作
def make_lmdb_for_single(image_path='Data_for_lmdb/apple.jpg', label='apple', lmdb_dir='lmdb_dir'):

    env = lmdb.open(lmdb_dir)
    cache = {}  # 存储键值对

    with open(image_path, 'rb') as f:
        # 读取图像文件的二进制格式数据
        image_bin = f.read()

    cache['image_000'] = image_bin
    cache['label_000'] = label

    txn = env.begin(write=True)
    for k, v in cache.items():
        if isinstance(v, bytes):
            # 图片类型为bytes
            txn.put(k.encode(), v)
        else:
            # 标签类型为str, 转为bytes
            txn.put(k.encode(), v.encode())  # 编码
    txn.commit()
    env.close()


# 制作一个的分割数据集
def make_lmdb_for_segmentation(image_path, label_path, txt_path, lmdb_dir):

    env = lmdb.open(lmdb_dir, 10487560*25)
    cache = {}  # 存储键值对

    with open(txt_path, 'r') as txt:
        lines = txt.readlines()
    i = 0
    for read_line in lines:
        image_file = read_line.split(',')[0]
        with open(image_path + image_file, 'rb') as f:
            # 读取图像文件的二进制格式数据
            image_bin = f.read()
        label_file = read_line.split(',')[1].replace('\n', '')
        with open(label_path + label_file, 'rb') as f:
            # 读取图像文件的二进制格式数据
            label_bin = f.read()
        cache['image_%04d' % i] = image_bin
        cache['label_%04d' % i] = label_bin
        i = i + 1

    txn = env.begin(write=True)
    for k, v in cache.items():
        if isinstance(v, bytes):
            # 图片类型为bytes
            txn.put(k.encode(), v)
    txn.commit()
    env.close()
    print("Solve")


# make_lmdb_for_segmentation(image_path=r'L:\ALASegmentationNets_v2\Data\Stage_4\train\img/',
#                            label_path=r'L:\ALASegmentationNets_v2\Data\Stage_4\train\mask/',
#                            txt_path=r'L:\ALASegmentationNets_v2\Data\Stage_4\train.txt',
#                            lmdb_dir=r'L:\ALASegmentationNets_v2\Data\Stage_4\Train_LMDB')






