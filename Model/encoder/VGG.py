from base import *


# model是网络对象
model = object


# ==========================================================================================
#                                Encoder
# ==========================================================================================

# 首先是最基础的VGG_Encoder，根据依靠的架构也各有不同

def VGG_Encoder(input_shape=(224, 224, 3), encoder_name='vgg16'):
    global model
    assert input_shape[0] % 32 == 0
    assert input_shape[1] % 32 == 0

    if encoder_name not in ['vgg11', 'vgg16', 'vgg19']:
        raise 'Please Choose model form [vgg11, vgg16, vgg19]'
    encoder_dict = {'vgg11': [1, 1, 2, 2, 2], 'vgg16': [2, 2, 3, 3, 3], 'vgg19': [2, 2, 4, 4, 4]}
    encoder_list = encoder_dict[encoder_name]

    # Creat Model
    input_layer = keras.layers.Input(shape=input_shape)
    x = input_layer
    # block1
    for i in range(1, encoder_list[0] + 1):
        x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv{}'.format(i))(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_maxpool1')(x)
    f1 = x

    # block2
    for i in range(1, encoder_list[1] + 1):
        x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv{}'.format(i))(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_maxpool1')(x)
    f2 = x

    # block3
    for i in range(1, encoder_list[2] + 1):
        x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv{}'.format(i))(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_maxpool1')(x)
    f3 = x

    # block4
    for i in range(1, encoder_list[3] + 1):
        x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv{}'.format(i))(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_maxpool1')(x)
    f4 = x

    # block4
    for i in range(1, encoder_list[4] + 1):
        x = keras.layers.Conv2D(1024, (3, 3), activation='relu', padding='same', name='block5_conv{}'.format(i))(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_maxpool1')(x)

    # 因为f5就是输出的x，所以没必要重新写了
    features = [f1, f2, f3, f4]

    model = keras.Model(input_layer, x)

    return model, features
