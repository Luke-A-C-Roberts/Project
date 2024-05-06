from utils import multi_layers

from tensorflow._api.v2.nn import local_response_normalization
from keras.models import Sequential
from keras.layers import (
    Conv2D,
    Dense,
    Flatten,
    MaxPooling2D,
    Lambda,
    Activation,
    Layer,
    Dropout,
    Resizing
)
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam
from keras import Input

from functools import partial

# AlexNet #####################################################################
def alexnet_pooling_layers(filters: int, kernel_size: int, name: str) -> list[Layer]:
    return [
        Conv2D(filters, kernel_size, strides=4, padding="same", name=f"{name}_conv2d"),
        Lambda(local_response_normalization, name=f"{name}_normalisation"),
        Activation("relu", name=f"{name}_activation"),
        MaxPooling2D(3, strides=2, name=f"{name}_maxpool"),
    ]


def alexnet_conv_layers(filters: int, kernel_size: int, name: int) -> list[Layer]:
    return [
        Conv2D(filters, kernel_size, strides=4, padding="same", name=f"{name}_conv2d"),
        Activation("relu", name=f"{name}_activation"),
    ]


def alexnet_dense_layers(units: int, name: str) -> list[Layer]:
    return [
        Dense(units, activation="relu", name=f"{name}_dense"),
        Dropout(0.5, name=f"{name}_dropout"),
    ]


# [1]
def build_alexnet(outputs: int) -> Sequential:
    alexnet = Sequential([
        Input(shape=(224, 224, 3), name="input"),
        *alexnet_pooling_layers(filters=96, kernel_size=11, name="pooling1"),
        *alexnet_pooling_layers(filters=256, kernel_size=5, name="pooling2"),
        *alexnet_conv_layers(filters=384, kernel_size=3, name="conv1"),
        *alexnet_conv_layers(filters=384, kernel_size=3, name="conv2"),
        *alexnet_conv_layers(filters=256, kernel_size=3, name="conv3"),
        Flatten(name="flatten"),
        *alexnet_dense_layers(units=4096, name="dense1"),
        *alexnet_dense_layers(units=4096, name="dense2"),
        Dense(outputs, activation="softmax"),
    ])

    alexnet.compile(
        loss=SparseCategoricalCrossentropy(from_logits=False),
        optimizer=Adam(),
        metrics=["accuracy"],
    )

    return alexnet


[1] # https://medium.com/swlh/alexnet-with-tensorflow-46f366559ce8
