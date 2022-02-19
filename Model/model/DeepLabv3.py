import keras.layers

import Model.encoder.ResNet
from Model.encoder import ResNet
from Model.model.ResNet import ResNetDecoder
from base import *

# model是网络对象
model = object


def DeepLabV3(input_shape=(224, 224, 3)):
    # 首先得获取到ResNet50的Backbone

    def Resnet50_backbone(input_shape=input_shape):
        backbone = ResNet.ResNet_Encoder(input_shape=input_shape, encoder_name='resnet50')
        input_layer = backbone.input
        output_layer = backbone.layers['block2']


Encoder = Model.encoder.ResNet.ResNet_Encoder(input_shape=(224, 224, 3), encoder_name='resnet50')
model = ResNetDecoder(Encoder, out_num=2)

