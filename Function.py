import os


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


get_txt()
