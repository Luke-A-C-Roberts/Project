from keras import Model
from keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Add,
    ZeroPadding2D,
    Layer,
    Activation,
    GlobalAveragePooling2D,
    Dense,
)
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam

from dataclasses import dataclass
from itertools import starmap
from typing import Callable

from resnet_specification import (
    LayerSpec,
    MultiLayerSpec,
    ResnetSpec,
    RESNET_SPECIFICATIONS,
)
from utils import *


# [1]
def conv_1(input_layer: Layer) -> Layer:
    layers: list[Layer] = [
        ZeroPadding2D(padding=pad(3), name="conv0_pad1"),
        Conv2D(filters=64, kernel_size=7, strides=2, name="conv0_conv"),
        BatchNormalization(name="conv0_norm"),
        ZeroPadding2D(padding=pad(1), name="conv0_pad2"),
        MaxPooling2D(pool_size=3, strides=2, name="conv0_pool"),
    ]
    return pipeline(layers, input_layer)


def conv_x(x: Layer, index: int, multilayer_spec: MultiLayerSpec) -> Layer:
    use_strides: bool = multilayer_spec.use_strides
    for repetition in range(multilayer_spec.repetitions):
        layer_name: str = "conv{0}_{1}".format(index + 1, repetition)
        shortcut_layers: list[Layer] = [
            Conv2D(
                filters=multilayer_spec.layer_specs[-1].filters,
                kernel_size=1,
                strides=(2 if use_strides else 1),
                name="{0}_shortcut_conv".format(layer_name),
            ),
            BatchNormalization(name="{0}_shortcut_norm".format(layer_name)),
        ]
        shortcut: Layer = pipeline(shortcut_layers, x)

        for layer_num, (kernel_size, filters) in enumerate(multilayer_spec):
            layers = [
                Conv2D(
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=(2 if use_strides else 1),
                    name="{0}_conv{1}".format(layer_name, layer_num),
                    padding="same",  # This is necessary because otherwise the output tensor with N×N input shape will be (N-1)×(N-1)
                ),
                BatchNormalization(name="{0}_norm{1}".format(layer_name, layer_num)),
            ]
            use_strides = False

            x = pipeline(layers, x)

            if layer_num == len(multilayer_spec) - 1:
                x = Add(name="{0}_add{1}".format(layer_name, layer_num))([shortcut, x])  # type: ignore (pylance can't type `Add`)

            x = Activation(
                activation="relu",
                name="{0}_activation{1}".format(layer_name, layer_num),
            )(x)  # type: ignore (pylance can't type `Add`)

    return x


def dense(x: Layer, index: int, output_classes: int) -> Layer:
    layer_name = "conv{0}".format(index)
    layers = [
        BatchNormalization(name="{0}_norm".format(layer_name)),
        Activation(activation="relu", name="{0}_activation".format(layer_name)),
        GlobalAveragePooling2D(name="{0}_pool".format(layer_name)),
        Dense(
            units=output_classes,
            activation="softmax",
            name="{0}_dense".format(layer_name),
        ),
    ]
    return pipeline(layers, x)


# https://arxiv.org/abs/1512.03385
def build_resnet(layers: int, output_classes: int) -> Model:
    resnet_nums = RESNET_SPECIFICATIONS.keys()
    if layers not in resnet_nums:
        raise ValueError(
            "{0} requires a layer number included in the list {1}".format(
                build_resnet, resnet_nums
            )
        )

    specification = RESNET_SPECIFICATIONS[layers]

    input_layer = Input(shape=(224, 224, 3))
    x = conv_1(input_layer)  # type: ignore
    for i, multilayer_spec in enumerate(specification):
        # ic(multilayer_spec)
        x = conv_x(x, i, multilayer_spec)
    output_layer = dense(x, len(specification) + 1, output_classes)

    resnet = Model(inputs=input_layer, outputs=output_layer)

    resnet.compile(
        loss=SparseCategoricalCrossentropy(from_logits=True),
        optimizer=Adam(lr=3e-4),
        metrics=["accuracy"],
    )

    return resnet

[1] # https://github.com/keras-team/keras/blob/master/keras/applications/resnet.py
