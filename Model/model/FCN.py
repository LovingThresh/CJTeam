from base import *
# model是网络对象
model = object


# FCN系列网络-FCN-8s，FCN-16s，FCN-32s
# 在经典的VGGNet的基础上，把VGG网络最后的全连接层全部去掉，换为卷积层
def FCN_32sDecoder(Encoder, feature_list=None, out_num=None):
    global model
    f1, f2, f3, f4 = feature_list
    Encoder_out_layer = Encoder.output

    # D_block1
    x = keras.layers.Conv2D(4096, (7, 7), strides=(1, 1), activation='relu', padding='same',
                            name='D_block1_conv1')(Encoder_out_layer)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Conv2D(4096, (1, 1), strides=(1, 1), padding='same', activation='relu',
                            name='D_block1_conv2')(x)
    x = keras.layers.Dropout(0.5)(x)

    # D_block2
    x = keras.layers.Conv2D(out_num, (1, 1), kernel_initializer='he_normal',
                            name='Seg_feats')(x)

    # D_block3
    x = keras.layers.Conv2DTranspose(out_num, (64, 64), strides=(32, 32), use_bias=False,
                                     name='D_block3_Tconv1')(x)

    model = keras.models.Model(inputs=Encoder.input, outputs=x, name='FCN_32s')

    return model


def FCN_16sDecoder(Encoder, feature_list=None, out_num=None):
    global model
    f1, f2, f3, f4, _ = feature_list
    Encoder_out_layer = Encoder.output

    # D_block1
    x = keras.layers.Conv2D(4096, (7, 7), strides=(1, 1), activation='relu', padding='same',
                            name='D_block1_conv1')(Encoder_out_layer)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Conv2D(4096, (1, 1), strides=(1, 1), padding='same', activation='relu',
                            name='D_block1_conv2')(x)
    x = keras.layers.Dropout(0.5)(x)

    # D_block2
    x = keras.layers.Conv2D(out_num, (1, 1), kernel_initializer='he_normal',
                            name='Seg_feats')(x)

    # D_block3
    x = keras.layers.Conv2DTranspose(out_num, (4, 4), (2, 2), use_bias=False,
                                     name='D_block3_Tconv1')(x)

    # D_block4
    x = keras.layers.Conv2DTranspose(out_num, (4, 4), strides=(2, 2), use_bias=False,
                                     name='D_block4_Tconv1')(x)

    # D_block5
    f3 = keras.layers.Conv2D(out_num, (1, 1), kernel_initializer='he_normal',
                             name='D_block5_conv1')(f3)
    x = keras.layers.Add(name='D_block5_add1')([x, f3])
    x = keras.layers.Conv2DTranspose(out_num, (16, 16), strides=(8, 8), use_bias=False,
                                     name='D_block5')(x)

    model = keras.models.Model(inputs=Encoder.input, outputs=x, name='FCN_8s')

    return model


def FCN_8sDecoder(Encoder, feature_list=None, out_num=None):
    global model
    f1, f2, f3, f4, _ = feature_list
    Encoder_out_layer = Encoder.output

    # D_block1
    x = keras.layers.Conv2D(4096, (7, 7), strides=(1, 1), activation='relu', padding='same',
                            name='D_block1_conv1')(Encoder_out_layer)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Conv2D(4096, (1, 1), strides=(1, 1), padding='same', activation='relu',
                            name='D_block1_conv2')(x)
    x = keras.layers.Dropout(0.5)(x)

    # D_block2
    x = keras.layers.Conv2D(out_num, (1, 1), kernel_initializer='he_normal',
                            name='Seg_feats')(x)

    # D_block3
    x = keras.layers.Conv2DTranspose(out_num, (4, 4), (2, 2), use_bias=False,
                                     name='D_block3_Tconv1')(x)

    # D_block4
    f4 = keras.layers.Conv2D(out_num, (1, 1), kernel_initializer='he_normal',
                             name='D_block4_conv1')(f4)
    x = keras.layers.Add(name='D_block4_add1')([x, f4])
    x = keras.layers.Conv2DTranspose(out_num, (4, 4), strides=(2, 2), use_bias=False,
                                     name='D_block4_Tconv1')(x)

    # D_block5
    f3 = keras.layers.Conv2D(out_num, (1, 1), kernel_initializer='he_normal',
                             name='D_block5_conv1')(f3)
    x = keras.layers.Add(name='D_block5_add1')([x, f3])
    x = keras.layers.Conv2DTranspose(out_num, (16, 16), strides=(8, 8), use_bias=False,
                                     name='D_block5')(x)

    model = keras.models.Model(inputs=Encoder.input, outputs=x, name='FCN_8s')

    return model
