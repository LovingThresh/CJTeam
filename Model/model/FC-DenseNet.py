#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/2/3
# @Author  : Liu Ye
# @File    : .py
# @IDE: PyCharm

import keras.optimizers
from keras import Input
from keras.layers import Conv2D, MaxPool2D, Conv2DTranspose, Activation, Dropout
from keras.layers import BatchNormalization
from keras.activations import relu
import tensorflow as tf
import cv2 as cv
import numpy as np


class FC_Densenet:

    def __init__(self, input_shape=(500, 500, 3), growth_rate=16, layers_per_block=(4, 5, 7), num_of_classes=2):
        self.input_shape = input_shape
        self.growth_rate = growth_rate
        self.layers_per_block = layers_per_block
        self.number_of_dense_blocks = len(self.layers_per_block)
        self.num_of_classes = num_of_classes

    @staticmethod
    def init_conv(input):
        conv_image = Conv2D(filters=48, kernel_size=(3, 3), strides=(1, 1), padding='SAME', name='init_conv')(input)
        return conv_image

    def dense_block_layer(self, input, conv_name, dropout_name):
        batch_norm_output = BatchNormalization()(input)
        relu_output = relu(batch_norm_output)
        conv_output = Conv2D(filters=self.growth_rate, kernel_size=(3, 3), strides=(1, 1), padding='SAME',
                             name=conv_name)(relu_output)
        conv_output = Dropout(0.2, name=dropout_name)(conv_output)

        return conv_output

    # dense_block_number=(4, 5, 7)
    def dense_block(self, input, dense_block_number, conv_name="", dropout_name=""):
        current_layers_count = self.layers_per_block[dense_block_number]
        all_layers_stack = []
        for layer in range(current_layers_count):
            conv_name = conv_name + str(layer+1)
            dropout_name = dropout_name + str(layer+1)
            layer_output = self.dense_block_layer(input=input, conv_name=conv_name, dropout_name=dropout_name)
            input = tf.concat([layer_output, input], axis=-1)
            all_layers_stack.append(layer_output)

        db_output = tf.concat(all_layers_stack, axis=-1)

        return db_output

    @staticmethod
    def transition_down(input, dense_block_number):
        batch_norm_output = BatchNormalization()(input)
        relu_output = relu(batch_norm_output)
        conv_output = Conv2D(filters=input.get_shape()[-1], kernel_size=(1, 1), strides=(1, 1), padding='SAME',
                             name='convtd_{}'.format(str(dense_block_number+1)))(relu_output)
        conv_output = Dropout(0.2, name='dropout_td_{}'.format(str(dense_block_number+1)))(conv_output)
        max_pool_output = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='SAME',
                                    name='maxpool_td_{}'.format(str(dense_block_number+1)))(conv_output)

        return max_pool_output

    @staticmethod
    def transition_up(input, dense_block_number):
        conv_output = Conv2DTranspose(filters=input.get_shape()[-1], kernel_size=(3, 3), strides=(2, 2), padding='SAME',
                                      name='convtu_{}'.format(str(dense_block_number+1)))(input)

        return conv_output

    def create_model(self):
        input_image = Input(shape=self.input_shape)

        all_concats = []

        # ############################################### ENCODER ##################################################
        input_encoder = self.init_conv(input_image)
        print("Data shape after init convolution: ")
        print(input_encoder.get_shape())
        print("\n")

        for db in range(self.number_of_dense_blocks-1):
            conv_name = "convdb_" + str(db+1) + "_layer_"
            dropout_name = "dropout_" + str(db+1) + "_layer_"
            db_output = self.dense_block(input=input_encoder, dense_block_number=db, conv_name=conv_name,
                                         dropout_name=dropout_name)

            concat_output = tf.concat([input_encoder, db_output], axis=-1, name='encoder_concat_{}'.format(db+1))
            all_concats.append(concat_output)
            input_encoder = self.transition_down(input=concat_output, dense_block_number=db)
            print("Data shape after DB+TD{}: ".format(db+1))
            print(input_encoder.get_shape())
            print("\n")

        output_encoder = input_encoder

        # ############################################# BOTTLENECK #################################################
        conv_name = "convdb_bottleneck" + "_layer_"
        dropout_name = "dropout_bottleneck" + "_layer_"
        output_bottleneck = self.dense_block(input=output_encoder, dense_block_number=self.number_of_dense_blocks-1,
                                             conv_name=conv_name, dropout_name=dropout_name)
        print("Data shape after Bottleneck: ")
        print(output_bottleneck.get_shape())
        print("\n")

        input_decoder = output_bottleneck

        # ############################################### DECODER ##################################################
        for db in range(self.number_of_dense_blocks-2, -1, -1):
            tu_output = self.transition_up(input=input_decoder, dense_block_number=db)
            concat_output = tf.concat([tu_output, all_concats[db]], axis=-1, name='decoder_concat_{}'.format(db+1))
            conv_name = "convdb_tu_" + str(db+1) + "_layer_"
            dropout_name = "dropout_tu_" + str(db+1) + "_layer_"
            input_decoder = self.dense_block(input=concat_output, dense_block_number=db, conv_name=conv_name,
                                             dropout_name=dropout_name)
            print("Data shape after TU+DB{}: ".format(db+1))
            print(concat_output.get_shape())
            print("\n")

        output_decoder = input_decoder

        # ############################################### SOFTMAX ##################################################
        conv_output = Conv2D(filters=self.num_of_classes, kernel_size=(1, 1), strides=(1, 1), padding='SAME',
                             name='conv_1_1')(output_decoder)
        segmentation_mask = Activation('softmax')(conv_output)
        print("Data shape of segmentation mask: ")
        print(segmentation_mask.get_shape())
        print("\n")

        model = keras.Model(inputs=input_image, outputs=segmentation_mask)

        return model


