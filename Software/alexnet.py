from tensorflow.nn import local_response_normalization

from keras.models import Sequential
from keras.layers import (
    Conv2D,
    Dense,
    MaxPooling2D,
    Flatten,
    Lambda,
    Activation,
    Layer,
    Dropout,
)
from keras.layers.experimental.preprocessing import Resizing
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam
from keras import Input

from utils import multi_layers


# AlexNet #####################################################################
def alexnet_pooling_layers(filters: int, kernel_size: int) -> list[Layer]:
    return [
        Conv2D(filters, kernel_size, strides=4, padding="same"),
        Lambda(local_response_normalization),
        Activation("relu"),
        MaxPooling2D(3, strides=2),
    ]


def alexnet_conv_layers(filters: int, kernel_size: int) -> list[Layer]:
    return [Conv2D(filters, kernel_size, strides=4, padding="same"), Activation("relu")]


def alexnet_dense_layers(units: int) -> list[Layer]:
    return [Dense(units, activation="relu"), Dropout(0.5)]


# https://medium.com/swlh/alexnet-with-tensorflow-46f366559ce8
def build_alex_net(x_train_shape: int) -> Sequential:
    alexnet = Sequential(
        [
            Resizing(
                height=224,
                width=224,
                interpolation="bilinear",
                input_shape=x_train_shape[1:],
            ),
            *alexnet_pooling_layers(filters=96, kernel_size=11),
            *alexnet_pooling_layers(filters=256, kernel_size=5),
            *multi_layers(alexnet_conv_layers, n=2, filters=384, kernel_size=3)
            * alexnet_conv_layers(filters=256, kernel_size=3),
            Flatten(),
            *multi_layers(alexnet_dense_layers, n=2, units=4096),
            Dense(10, activation="softmax"),
        ]
    )

    alexnet.compile(
        loss=SparseCategoricalCrossentropy(from_logits=True),
        optimizer=Adam(lr=3e-4),
        metrics=["accuracy"],
    )

    return alexnet
