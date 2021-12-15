import os
import numpy as np


# 从指定路径生成训练文本

def get_txt():
    a_path = r'L:\crack_segmentation_dataset\crack_segmentation_dataset/'
    file_list = ['train', 'val', 'test']
    for file in file_list:
        mask_list = os.listdir(a_path + '/' + file + '/masks/')
        with open(r'L:\crack_segmentation_dataset\crack_segmentation_dataset/' + '{}.txt'.format(file), 'w') as f:
            for mask in mask_list[:-1]:
                if '_mask' in mask:
                    a = mask[:-9] + '.jpg' + ',' + mask + '\n'
                elif '_mask' not in mask:
                    a = mask[:-4] + '.jpg' + ',' + mask + '\n'
                f.write(a)
            mask = mask_list[-1]
            if '_mask' in mask_list[-1]:
                a = mask[:-9] + '.jpg' + ',' + mask
            elif '_mask' not in mask_list[-1]:
                a = mask[:-4] + '.jpg' + ',' + mask
            f.write(a)


# 获得数据集的均值与方差来初始化数据集
def get_mean_and_std_of_Dataset(filepath):
    filepath = r'Z:\pycharm_projects\ssd\VOC2007\reshape'  # 数据集目录
    pathDir = os.listdir(filepath)

    R_channel = 0
    G_channel = 0
    B_channel = 0
    for idx in range(len(pathDir)):
        filename = pathDir[idx]
        img = imread(os.path.join(filepath, filename)) / 255.0
        R_channel = R_channel + np.sum(img[:, :, 0])
        G_channel = G_channel + np.sum(img[:, :, 1])
        B_channel = B_channel + np.sum(img[:, :, 2])

    num = len(pathDir) * 512 * 512  # 这里（512,512）是每幅图片的大小，所有图片尺寸都一样
    R_mean = R_channel / num
    G_mean = G_channel / num
    B_mean = B_channel / num

    R_channel = 0
    G_channel = 0
    B_channel = 0
    for idx in range(len(pathDir)):
        filename = pathDir[idx]
        img = imread(os.path.join(filepath, filename)) / 255.0
        R_channel = R_channel + np.sum((img[:, :, 0] - R_mean) ** 2)
        G_channel = G_channel + np.sum((img[:, :, 1] - G_mean) ** 2)
        B_channel = B_channel + np.sum((img[:, :, 2] - B_mean) ** 2)

    R_var = np.sqrt(R_channel / num)
    G_var = np.sqrt(G_channel / num)
    B_var = np.sqrt(B_channel / num)
    print("R_mean is %f, G_mean is %f, B_mean is %f" % (R_mean, G_mean, B_mean))
    print("R_var is %f, G_var is %f, B_var is %f" % (R_var, G_var, B_var))



