from base import *
from Layer import Layer
# model是网络对象
model = object


# DeepLab系列分割网络
# DeepLab的Motivation: ①上采样分辨率低;②空间不敏感
# DeepLab的特点Atrous Convolution;MSc;CRF;LargeFOV;只下采样八倍
def DeepLabV1_Encoder(input_shape=(224, 224, 3)):
    global model
    assert input_shape[0] % 8 == 0
    assert input_shape[1] % 8 == 0

    input_layer = keras.layers.Input(shape=input_shape)

    # variable_with_weight_loss
    def variable_with_weight_loss(shape, stddev, w1):
        var = tf.Variable(tf.compat.v1.truncated_normal(shape, stddev=stddev))
        if w1 is not None:
            weight_loss = tf.multiply(tf.nn.l2_loss(var), w1)
            tf.compat.v1.add_to_collection('losses', weight_loss)
        return var

    # block1
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_layer)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block1_maxpool1')(x)
    f1 = x

    # block2
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block2_maxpool1')(x)
    f2 = x

    # block3
    x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block3_maxpool1')(x)
    f3 = x

    # block4
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = keras.layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same', name='block4_maxpool1')(x)
    f4 = x

    # block5
    # 经过block5之后，模型的输出为W/32， H/32， 512

    x = Layer.DilatedConv2D(sizes=3, rate=2, out_channel=512, padding='SAME', name='block5_Aconv1')(x)
    x = keras.layers.ReLU(name='block5_relu1')(x)
    x = Layer.DilatedConv2D(sizes=3, rate=2, out_channel=512, padding='SAME', name='block5_Aconv2')(x)
    x = keras.layers.ReLU(name='block5_relu2')(x)
    x = Layer.DilatedConv2D(sizes=3, rate=2, out_channel=512, padding='SAME', name='block5_Aconv3')(x)
    x = keras.layers.ReLU(name='block5_relu3')(x)
    x = keras.layers.MaxPooling2D((3, 3), strides=(1, 1), padding='SAME', name='block5_maxpool1')(x)
    x = keras.layers.AveragePooling2D((3, 3), strides=(1, 1), padding='SAME', name='block5_avepool1')(x)

    features = [f1, f2, f3, f4]

    model = keras.Model(input_layer, x)

    return model, features


# Total params: 17,599,810
def DeepLabV1_Decoder_with_FOV(Encoder, feature_list=None, out_num=None):
    global model
    f1, f2, f3, f4 = feature_list
    Encoder_out_layer = Encoder.output

    # D_block1
    x = Layer.DilatedConv2D(sizes=3, rate=12, out_channel=512, padding='SAME', name='D_block1_Aconv1')(
        Encoder_out_layer)
    x = keras.layers.Dropout(0.5)(x)

    # D_block2
    x = keras.layers.Conv2D(1024, (1, 1), strides=1, activation='relu', name='D_block2_conv1')(x)
    x = keras.layers.Dropout(0.5)(x)

    # D_block3
    x = keras.layers.Conv2D(out_num, (1, 1), strides=1, activation='relu', name='D_block3_conv1')(x)

    # Upsample
    for i in range(3):
        x = keras.layers.UpSampling2D((2, 2), name='Upsample{}'.format(i + 1))(x)

    model = keras.models.Model(inputs=Encoder.input, outputs=x, name='DeepLabV1')

    return model


# Total params: 18,793,676
def DeepLabV1_Decoder_with_FOV_MSc(Encoder, feature_list=None, out_num=None):
    global model
    f1, f2, f3, f4 = feature_list
    Encoder_out_layer = Encoder.output

    # D_block1
    x = Layer.DilatedConv2D(sizes=3, rate=12, out_channel=512, padding='SAME', name='D_block1_Aconv1')(
        Encoder_out_layer)
    x = keras.layers.Dropout(0.5)(x)

    # D_block2
    x = keras.layers.Conv2D(1024, (1, 1), strides=1, activation='relu', name='D_block2_conv1')(x)
    x = keras.layers.Dropout(0.5)(x)

    # D_block3
    x = keras.layers.Conv2D(out_num, (1, 1), strides=1, activation='relu', name='D_block3_conv1')(x)

    # Branch
    def DownSample_for_MSc(k_size, stride, out_channel):
        MSc_block = keras.Sequential([
            keras.layers.Conv2D(128, (k_size, k_size), strides=stride, padding='same', activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Conv2D(128, (1, 1), strides=1, padding='same', activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Conv2D(out_channel, (1, 1), strides=1, padding='same')
        ])

        return MSc_block

    x0 = DownSample_for_MSc(3, 8, out_num)(Encoder.input)
    x1 = DownSample_for_MSc(3, 4, out_num)(f1)
    x2 = DownSample_for_MSc(3, 2, out_num)(f2)
    x3 = DownSample_for_MSc(3, 1, out_num)(f3)
    x4 = DownSample_for_MSc(3, 1, out_num)(f4)

    x = keras.layers.Add()([x, x0, x1, x2, x3, x4])

    # Upsample
    for i in range(3):
        x = keras.layers.UpSampling2D((2, 2), name='Upsample{}'.format(i + 1))(x)

    model = keras.models.Model(inputs=Encoder.input, outputs=x, name='DeepLabV1_MSc')

    return model
