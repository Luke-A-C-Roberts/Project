from keras import Model
from keras.api.layers import (
    Activation,
    AveragePooling2D,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Dense,
    GlobalAveragePooling2D,
    Input,
    Layer,
    MaxPooling2D,
    ZeroPadding2D,
)
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam

from dataclasses import dataclass
from itertools import starmap
from typing import Callable, KeysView

from densenet_specification import (
    LayerSpec,
    MultiLayerSpec,
    DensenetSpec,
    DENSENET_SPECIFICATIONS,
)
from utils import *

from more_itertools import intersperse


# placeholder class for transition blocks used in densenet construction
class TransitionBlock: pass


# [1]
def densenet_conv_0(input_layer: Layer) -> Layer:
    layers: list[Layer] = [
        ZeroPadding2D(padding=pad(3), name="conv0_pad1"),
        Conv2D(filters=64, kernel_size=7, strides=2, name="conv0_conv"),
        BatchNormalization(name="conv0_norm"),
        Activation(activation="relu", name="conv0_activation"),
        ZeroPadding2D(padding=pad(1), name="conv0_pad2"),
        MaxPooling2D(pool_size=3, strides=2, name="conv0_pool"),
    ]
    return pipeline(layers, input_layer)


def densenet_conv_x(x: Layer, index: int, multilayer_spec: MultiLayerSpec) -> Layer:
    for repetition in range(multilayer_spec.repetitions):
        layer_name: str = f"conv{index + 1}_{repetition}"
        shortcut: Layer = x
        for layer_num, (kernel_size, filters) in enumerate(multilayer_spec):
            layers: list[Layer] = [
                BatchNormalization(name=f"{layer_name}_norm{layer_num}"),
                Activation(activation="relu", name=f"{layer_name}_activation{layer_num}"),
                Conv2D(
                    filters=filters,
                    kernel_size=kernel_size,
                    name=f"{layer_name}_{layer_num}",
                    padding="same",
                    use_bias=False
                )
            ]
            x = pipeline(layers, x)
        x = Concatenate(name=f"{layer_name}_concatinate")([x, shortcut])

    return x
        

def densenet_trans_block(x: Layer, index: int) -> Layer:
    layer_name: str = f"trans{index + 1}"
    layers: list[Layer] = [
        BatchNormalization(name=f"{layer_name}_norm"),
        Activation(activation="relu", name=f"{layer_name}_activation"),
        Conv2D(
            filters=x.shape[3] // 2,
            kernel_size=1,
            name=f"{layer_name}_conv",
            use_bias=False
        ),
        AveragePooling2D(pool_size=2, strides=2, name=f"{layer_name}_pool")
    ]
   
    return pipeline(layers, x)


def dense(x: Layer, index: int, output_classes: int) -> Layer:
    layers: list[Layer] = [
        BatchNormalization(name="dense_norm"),
        Activation(activation="relu", name=f"dense_activation"),
        GlobalAveragePooling2D(name="dense_pool"),
        Dense(output_classes, activation="softmax", name="output"),
    ]
    return pipeline(layers, x)


# [1]
def build_densenet(layers: int, output_classes: int) -> Model:
    densenet_nums: KeysView[int] = DENSENET_SPECIFICATIONS.keys()
    if layers not in densenet_nums:
        raise ValueError(
            "{0} requires a layer number included in the list {1}".format(
                build_densenet, densenet_nums
            )
        )

    specification: DensenetSpec = DENSENET_SPECIFICATIONS[layers]
    input_layer = Input(shape=(224, 224, 3), name="input")
    
    x: Layer = densenet_conv_0(input_layer)

    for i, multilayer_spec in enumerate(intersperse(TransitionBlock, specification)):
        if multilayer_spec == TransitionBlock:
            x = densenet_trans_block(x, i)
        else:
            x = densenet_conv_x(x, i, multilayer_spec)

    output_layer: Layer = dense(x, len(specification) + 1, output_classes)

    densenet = Model(inputs=input_layer, outputs=output_layer)

    densenet.compile(
        loss=SparseCategoricalCrossentropy(from_logits=False),
        optimizer=Adam(),
        metrics=["accuracy"],
    )

    return densenet


[1] # https://arxiv.org/abs/1608.06993
