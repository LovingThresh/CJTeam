import keras.layers

from base import *

# model是网络对象
model = object


# ==========================================================================================
#                                Encoder
# ==========================================================================================
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

    def call(self, inputs, training=False, **kwargs):
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


def ResNet_Encoder(input_shape=(224, 224, 3), encoder_name='resnet18'):
    global model
    assert input_shape[0] % 32 == 0
    assert input_shape[1] % 32 == 0

    if encoder_name not in ['res18', 'res34', 'res50', 'res101', 'res152']:
        raise 'Please Choose model form [res18, res34, res50, res101, res152]'
    encoder_dict = {'res18': [2, 2, 2, 2], 'res34': [3, 4, 6, 3],
                    'res50': [3, 4, 6, 3], 'res101': [3, 4, 23, 3],
                    'res152': [3, 8, 36, 3]}

    encoder_num = encoder_dict[encoder_name]

    def _make_layer(block, in_channel, channel, block_num, name, strides=1):
        downsample = None
        if strides != 1 or in_channel != channel * block.expansion:
            downsample = keras.Sequential([
                keras.layers.Conv2D(channel * block.expansion, kernel_size=1, strides=strides,
                                    use_bias=False, name="conv1"),
                keras.layers.BatchNormalization(momentum=0.9, epsilon=1.001e-5, name="BatchNorm")
            ], name="shortcut")

        layers_list = [block(channel, downsample=downsample, strides=strides, name="unit_1")]

        for index in range(1, block_num):
            layers_list.append(block(channel, name="unit_" + str(index + 1)))

        return keras.Sequential(layers_list, name=name)

    def _resnet(block, blocks_num, im_width=224, im_height=224):
        global model
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

        predict = x
        model = keras.models.Model(inputs=input_image, outputs=predict)

        return model

    model = _resnet(BasicBlock, encoder_num, im_width=input_shape[0], im_height=input_shape[1])

    return model
