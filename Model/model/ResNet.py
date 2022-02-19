from base import *
from Layer import Layer
# model是网络对象
model = object


def ResNetDecoder(Encoder, out_num):
    global model
    n_upsample = 5
    Encoder_out_layer = Encoder.output

    x = Encoder_out_layer
    for i in range(n_upsample - 1):
        dim = Encoder_out_layer.shape[-1]
        dim = np.int(dim / 2)

        # y = Layer.DilatedConv2D(k_size=3, rate=2, out_channel=dim, padding='SAME', name='D_block1_Aconv{}'.format(i + 1))(x)
        # y = keras.layers.Conv2DTranspose(dim, (3, 3), strides=2, padding='same', use_bias=False)(y)
        # y = keras.layers.BatchNormalization()(y)
        # y = keras.layers.ReLU()(y)
        # y = keras.layers.Softmax()(y)

        x = keras.layers.Conv2DTranspose(dim, (3, 3), strides=2, padding='same', use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        # x = keras.layers.Multiply()([x, y])

    dim = out_num

    # y = Layer.DilatedConv2D(k_size=3, rate=2, out_channel=dim, padding='SAME', name='D_block1_Aconv5')(x)
    # y = keras.layers.Conv2DTranspose(dim, (3, 3), strides=2, padding='same', use_bias=False)(y)
    # y = keras.layers.BatchNormalization()(y)
    # y = keras.layers.ReLU()(y)
    # y = keras.layers.Softmax()(y)

    x = keras.layers.Conv2DTranspose(dim, (3, 3), strides=2, padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    # x = keras.layers.Multiply()([x, y])

    model = keras.models.Model(inputs=Encoder.input, outputs=x, name='ResNet-Segmentation')

    return model

