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
from dataclasses import dataclass
from itertools import starmap
from typing import Callable, Iterable, Iterator
from utils import *

# from icecream import ic


@dataclass(frozen=True)
class LayerSpec(Iterable):
    kernel_size: int
    filters: int

    def __iter__(self) -> Iterator:
        return iter([self.kernel_size, self.filters])


@dataclass(frozen=True)
class MultiLayerSpec(Iterable):
    layer_specs: list[LayerSpec]
    repetitions: int
    use_strides: bool = True

    def __iter__(self) -> Iterator:
        return iter(map(tuple, self.layer_specs))

    def __len__(self) -> int:
        return len(self.layer_specs)


@dataclass(frozen=True)
class ResnetSpec(Iterable):
    multi_layer_specs: list[MultiLayerSpec]

    def __iter__(self) -> Iterator:
        return iter(self.multi_layer_specs)

    def __len__(self) -> int:
        return len(self.multi_layer_specs)


RESNET_SPECIFICATIONS: dict[int, ResnetSpec] = {
    18: ResnetSpec(
        [
            MultiLayerSpec([LayerSpec(3, 64), LayerSpec(3, 64)], 2, False),
            MultiLayerSpec([LayerSpec(3, 128), LayerSpec(3, 128)], 2),
            MultiLayerSpec([LayerSpec(3, 256), LayerSpec(3, 256)], 2),
            MultiLayerSpec([LayerSpec(3, 512), LayerSpec(3, 512)], 2),
        ]
    ),
    34: ResnetSpec(
        [
            MultiLayerSpec([LayerSpec(3, 64), LayerSpec(3, 64)], 3, False),
            MultiLayerSpec([LayerSpec(3, 128), LayerSpec(3, 128)], 4),
            MultiLayerSpec([LayerSpec(3, 256), LayerSpec(3, 256)], 6),
            MultiLayerSpec([LayerSpec(3, 512), LayerSpec(3, 512)], 3),
        ]
    ),
    50: ResnetSpec(
        [
            MultiLayerSpec(
                [LayerSpec(1, 64), LayerSpec(3, 64), LayerSpec(1, 256)], 3, False
            ),
            MultiLayerSpec(
                [LayerSpec(1, 128), LayerSpec(3, 128), LayerSpec(1, 512)], 4
            ),
            MultiLayerSpec(
                [LayerSpec(1, 256), LayerSpec(3, 256), LayerSpec(1, 1024)], 6
            ),
            MultiLayerSpec(
                [LayerSpec(1, 512), LayerSpec(3, 512), LayerSpec(1, 2048)], 3
            ),
        ]
    ),
    101: ResnetSpec(
        [
            MultiLayerSpec(
                [LayerSpec(1, 64), LayerSpec(3, 64), LayerSpec(1, 256)], 3, False
            ),
            MultiLayerSpec(
                [LayerSpec(1, 128), LayerSpec(3, 128), LayerSpec(1, 512)], 4
            ),
            MultiLayerSpec(
                [LayerSpec(1, 256), LayerSpec(3, 256), LayerSpec(1, 1024)], 23
            ),
            MultiLayerSpec(
                [LayerSpec(1, 512), LayerSpec(3, 512), LayerSpec(1, 2048)], 3
            ),
        ]
    ),
    152: ResnetSpec(
        [
            MultiLayerSpec(
                [LayerSpec(1, 64), LayerSpec(3, 64), LayerSpec(1, 256)], 3, False
            ),
            MultiLayerSpec(
                [LayerSpec(1, 128), LayerSpec(3, 128), LayerSpec(1, 512)], 8
            ),
            MultiLayerSpec(
                [LayerSpec(1, 256), LayerSpec(3, 256), LayerSpec(1, 1024)], 36
            ),
            MultiLayerSpec(
                [LayerSpec(1, 512), LayerSpec(3, 512), LayerSpec(1, 2048)], 3
            ),
        ]
    ),
}


# https://github.com/keras-team/keras/blob/master/keras/applications/resnet.py
def conv_1(input_layer: Layer) -> Layer:
    layers = [
        ZeroPadding2D(padding=pad(3), name="conv0_pad1"),
        Conv2D(filters=64, kernel_size=7, strides=2, name="conv0_conv"),
        BatchNormalization(name="conv0_norm"),
        ZeroPadding2D(padding=pad(1), name="conv0_pad2"),
        MaxPooling2D(pool_size=3, strides=2, name="conv0_pool"),
    ]
    return pipeline(layers, input_layer)


def conv_x(x: Layer, index: int, multilayer_spec: MultiLayerSpec) -> Layer:
    use_strides = multilayer_spec.use_strides
    for repetition in range(multilayer_spec.repetitions):
        layer_name = "conv{0}_{1}".format(index + 1, repetition)
        shortcut_layers = [
            Conv2D(
                filters=multilayer_spec.layer_specs[-1].filters,
                kernel_size=1,
                strides=(2 if use_strides else 1),
                name="{0}_shortcut_conv".format(layer_name),
            ),
            BatchNormalization(name="{0}_shortcut_norm".format(layer_name)),
        ]
        shortcut = pipeline(shortcut_layers, x)

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
            )(
                x
            )  # type: ignore (pylance can't type `Add`)

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
    return Model(inputs=input_layer, outputs=output_layer)


# resnet(34, 10).summary()
