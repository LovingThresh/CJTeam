# 这个文件创建的目的是为了复现(复制)目前比较主流的分割算法
# 以作为参照，真实地了解目前架构的性能
import numpy as np

import tensorflow as tf
import tensorflow.keras as keras
import Layer

pretrained_url = "https://github.com/fchollet/deep-learning-models/" \
                 "releases/download/v0.1/" \
                 "vgg16_weights_th_dim_ordering_th_kernels_notop.h5"
# model是网络对象
model = object


# ==========================================================================================
#                                Encoder
# ==========================================================================================

# 首先是最基础的VGG16_Encoder，根据依靠的架构也各有不同
def VGG_16_Encoder(input_shape=(224, 224, 3), pretrained=False):
    global model
    assert input_shape[0] % 32 == 0
    assert input_shape[1] % 32 == 0

    input_layer = keras.layers.Input(shape=input_shape)

    # block1
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_layer)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_maxpool1')(x)
    f1 = x

    # block2
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_maxpool1')(x)
    f2 = x

    # block3
    x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_maxpool1')(x)
    f3 = x

    # block4
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_maxpool1')(x)
    f4 = x

    # block5
    # 经过block5之后，模型的输出为W/32， H/32， 512
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_maxpool1')(x)

    features = [f1, f2, f3, f4]

    if pretrained == 'imagenet':
        VGG_Weights_path = keras.utils.get_file(pretrained_url.split("/")[-1], pretrained_url)
        model = keras.Model(inputs=input_layer, outputs=x) \
            .load_weights(VGG_Weights_path, by_name=True, skip_mismatch=True)
    if not pretrained:
        model = keras.Model(input_layer, x)

    return model, features


# ResNetEncoder使用类的形式来写
class BasicBlock(keras.layers.Layer):
    expansion = 1

    def __init__(self, out_channel, strides=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__(**kwargs)
        self.conv1 = keras.layers.Conv2D(out_channel, kernel_size=3, strides=strides,
                                         padding="SAME", use_bias=False)
        self.bn1 = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        # -----------------------------------------
        self.conv2 = keras.layers.Conv2D(out_channel, kernel_size=3, strides=1,
                                         padding="SAME", use_bias=False)
        self.bn2 = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        # -----------------------------------------
        self.downsample = downsample
        self.relu = keras.layers.ReLU()
        self.add = keras.layers.Add()

    def call(self, inputs, training=False):
        identity = inputs
        if self.downsample is not None:
            identity = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        x = self.add([identity, x])
        x = self.relu(x)

        return x


class Bottleneck(keras.layers.Layer):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4

    def __init__(self, out_channel, strides=1, downsample=None, **kwargs):
        super(Bottleneck, self).__init__(**kwargs)
        self.conv1 = keras.layers.Conv2D(out_channel, kernel_size=1, use_bias=False, name="conv1")
        self.bn1 = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv1/BatchNorm")
        # -----------------------------------------
        self.conv2 = keras.layers.Conv2D(out_channel, kernel_size=3, use_bias=False,
                                         strides=strides, padding="SAME", name="conv2")
        self.bn2 = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv2/BatchNorm")
        # -----------------------------------------
        self.conv3 = keras.layers.Conv2D(out_channel * self.expansion, kernel_size=1, use_bias=False, name="conv3")
        self.bn3 = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv3/BatchNorm")
        # -----------------------------------------
        self.relu = keras.layers.ReLU()
        self.downsample = downsample
        self.add = keras.layers.Add()

    def call(self, inputs, training=False):
        identity = inputs
        if self.downsample is not None:
            identity = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)

        x = self.add([x, identity])
        x = self.relu(x)

        return x


def _make_layer(block, in_channel, channel, block_num, name, strides=1):
    downsample = None
    if strides != 1 or in_channel != channel * block.expansion:
        downsample = keras.Sequential([
            keras.layers.Conv2D(channel * block.expansion, kernel_size=1, strides=strides,
                                use_bias=False, name="conv1"),
            keras.layers.BatchNormalization(momentum=0.9, epsilon=1.001e-5, name="BatchNorm")
        ], name="shortcut")

    layers_list = []
    layers_list.append(block(channel, downsample=downsample, strides=strides, name="unit_1"))

    for index in range(1, block_num):
        layers_list.append(block(channel, name="unit_" + str(index + 1)))

    return keras.Sequential(layers_list, name=name)


def _resnet(block, blocks_num, im_width=224, im_height=224, num_classes=1000, include_top=False):
    # tensorflow中的tensor通道排序是NHWC
    # (None, 224, 224, 3)
    input_image = keras.layers.Input(shape=(im_height, im_width, 3), dtype="float32")
    x = keras.layers.Conv2D(filters=64, kernel_size=7, strides=2,
                            padding="SAME", use_bias=False, name="conv1")(input_image)
    x = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv1/BatchNorm")(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.MaxPool2D(pool_size=3, strides=2, padding="SAME")(x)

    x = _make_layer(block, x.shape[-1], 64, blocks_num[0], name="block1")(x)
    x = _make_layer(block, x.shape[-1], 128, blocks_num[1], strides=2, name="block2")(x)
    x = _make_layer(block, x.shape[-1], 256, blocks_num[2], strides=2, name="block3")(x)
    x = _make_layer(block, x.shape[-1], 512, blocks_num[3], strides=2, name="block4")(x)

    if include_top:
        x = keras.layers.GlobalAvgPool2D()(x)  # pool + flatten
        x = keras.layers.Dense(num_classes, name="logits")(x)
        predict = keras.layers.Softmax()(x)
    else:
        predict = x

    model = keras.models.Model(inputs=input_image, outputs=predict)

    return model


def resnet34(im_width=224, im_height=224, num_classes=1000, include_top=False):
    return _resnet(BasicBlock, [3, 4, 6, 3], im_width, im_height, num_classes, include_top)


def resnet50(im_width=224, im_height=224, num_classes=1000, include_top=False):
    return _resnet(Bottleneck, [3, 4, 6, 3], im_width, im_height, num_classes, include_top)


def resnet101(im_width=224, im_height=224, num_classes=1000, include_top=False):
    return _resnet(Bottleneck, [3, 4, 23, 3], im_width, im_height, num_classes, include_top)


# ==========================================================================================
#                                Decoder
# ==========================================================================================

# 首先是分类器的Decoder，其对应的就是VGG16网络
def ClassificationDecoder(Encoder, num_class: int):
    global model
    Encoder_out_layer = Encoder.output
    x = keras.layers.Dense(4096, activation='relu')(Encoder_out_layer)
    x = keras.layers.Dense(4096, activation='relu')(x)
    x = keras.layers.Dense(num_class, activation='relu')(x)

    x = keras.layers.Softmax()(x)

    model = keras.Model(inputs=Encoder.input, outputs=x)

    return model


# SegmentationDecoder 对应的是U-Net的写法
def SegmentationDecoder(Encoder, feature_list=None, out_num=None):
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


# FCN系列网络-FCN-8s，FCN-16s，FCN-32s
# 在经典的VGGNet的基础上，把VGG网络最后的全连接层全部去掉，换为卷积层
def FCN_32sEncoder(Encoder, feature_list=None, out_num=None):
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


def FCN_8sEncoder(Encoder, feature_list=None, out_num=None):
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

    x = Layer.DilatedConv2D(k_size=3, rate=2, out_channel=512, padding='SAME', name='block5_Aconv1')(x)
    x = keras.layers.ReLU(name='block5_relu1')(x)
    x = Layer.DilatedConv2D(k_size=3, rate=2, out_channel=512, padding='SAME', name='block5_Aconv2')(x)
    x = keras.layers.ReLU(name='block5_relu2')(x)
    x = Layer.DilatedConv2D(k_size=3, rate=2, out_channel=512, padding='SAME', name='block5_Aconv3')(x)
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
    x = Layer.DilatedConv2D(k_size=3, rate=12, out_channel=512, padding='SAME', name='D_block1_Aconv1')(
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
    x = Layer.DilatedConv2D(k_size=3, rate=12, out_channel=512, padding='SAME', name='D_block1_Aconv1')(
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

# ==========================================================================================
#                                Test
# ==========================================================================================
Encoder = resnet34(512, 512)
model = ResNetDecoder(Encoder=Encoder, out_num=2)
model.summary()
