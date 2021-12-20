from base import *
# model是网络对象
model = object


# SegmentationDecoder 对应的是U-Net的写法
def U_NetDecoder(Encoder, feature_list=None, out_num=None):
    global model
    f1, f2, f3, f4 = feature_list
    Encoder_out_layer = Encoder.output

    # transpose_block1
    x = keras.layers.Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same',
                            name='transpose_block1_conv1')(Encoder_out_layer)
    x = keras.layers.Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same',
                            name='transpose_block1_conv2')(x)
    x = keras.layers.Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same',
                            name='transpose_block1_conv3')(x)
    x = keras.layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), activation='relu', padding='same',
                                     name='transpose_block1_Tconv1')(x)

    # transpose_block2
    x = keras.layers.concatenate([x, f4], axis=-1)
    x = keras.layers.Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same',
                            name='transpose_block2_conv1')(x)
    x = keras.layers.Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same',
                            name='transpose_block2_conv2')(x)
    x = keras.layers.Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same',
                            name='transpose_block2_conv3')(x)
    x = keras.layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), activation='relu', padding='same',
                                     name='transpose_block2_Tconv1')(x)

    # transpose_block3
    x = keras.layers.concatenate([x, f3], axis=-1)
    x = keras.layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same',
                            name='transpose_block3_conv1')(x)
    x = keras.layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same',
                            name='transpose_block3_conv2')(x)
    x = keras.layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same',
                            name='transpose_block3_conv3')(x)
    x = keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), activation='relu', padding='same',
                                     name='transpose_block3_Tconv1')(x)

    # transpose_block4
    x = keras.layers.concatenate([x, f2], axis=-1)
    x = keras.layers.Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same',
                            name='transpose_block4_conv1')(x)
    x = keras.layers.Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same',
                            name='transpose_block4_conv2')(x)
    x = keras.layers.Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same',
                            name='transpose_block4_conv3')(x)
    x = keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu', padding='same',
                                     name='transpose_block4_Tconv1')(x)

    # transpose_block5
    x = keras.layers.concatenate([x, f1], axis=-1)
    x = keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same',
                            name='transpose_block5_conv1')(x)
    x = keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same',
                            name='transpose_block5_conv2')(x)
    x = keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same',
                            name='transpose_block5_conv3')(x)
    x = keras.layers.Conv2DTranspose(out_num, (3, 3), strides=(2, 2), activation='relu', padding='same',
                                     name='transpose_block5_Tconv1')(x)

    model = keras.Model(inputs=Encoder.input, outputs=x, name='U-Net')

    return model
