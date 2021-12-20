from base import *
# model是网络对象
model = object


# ==========================================================================================
#                            ClassificationDecoder
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

