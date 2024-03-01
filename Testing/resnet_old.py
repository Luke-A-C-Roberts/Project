from keras import Model, Input
from keras.layers import (
    Add,
    Conv2D,
    BatchNormalization,
    Dense,
    MaxPooling2D,
    Layer,
)
from dataclasses import dataclass
from utils import call_reduce
from enum import Enum


# https://towardsdatascience.com/the-annotated-resnet-50-a6c536034758
def conv_1() -> Model:
    layer_input = Input(shape=(244, 244))
    layers = [
        Input(shape=(244, 244)),
        Conv2D(filters=64, kernel_size=7, stride=2),
        MaxPooling2D(pool_size=(3, 3), strides=2),
    ]
    output = call_reduce(layers, layer_input)
    return Model(inputs=layer_input, outputs=output, name="conv1")


def resnet_small_conv_layer(filters: int, final_batch: bool = True) -> list[Layer]:
    layers = [
        Conv2D(filters, kernel_size=3, activation="relu"),
        BatchNormalization(),
        Conv2D(filters, kernel_size=3, activation="relu"),
    ]
    if final_batch:
        layers.append(BatchNormalization())
    return layers


def resnet_large_conv_layer(filters: list[int], final_batch: bool = True) -> list[Layer]:
    layers = [
        Conv2D(filters[0], kernel_size=1, activation="relu"),
        BatchNormalization(),
        Conv2D(filters[1], kernel_size=3, activation="relu"),
        BatchNormalization(),
        Conv2D(filters[2], kernel_size=1, activation="relu"),
    ]
    if final_batch:
        layers.append(BatchNormalization())
    return layers


class ModelSize(Enum):
    small = 0
    large = 1


@dataclass(frozen=True)
class FilterSpecification(object):
    repetitions: list[int]
    filter_sizes: list[int] | list[list[int]]


@dataclass(frozen=True)
class LayerArchitecture(object):
    layers: int
    size: ModelSize
    filter_spec: FilterSpecification


LAYER_ARCHITECTURES: list[LayerArchitecture] = [
    LayerArchitecture(
        layers=18,
        size=ModelSize.small,
        filter_spec=FilterSpecification(repetitions=[2, 2, 2, 2], filter_sizes=[64, 128, 256, 512]),
    ),
    LayerArchitecture(
        layers=34,
        size=ModelSize.small,
        filter_spec=FilterSpecification(repetitions=[3, 4, 6, 3], filter_sizes=[64, 128, 256, 512]),
    ),
    LayerArchitecture(
        layers=50,
        size=ModelSize.large,
        filter_spec=FilterSpecification(
            repetitions=[3, 4, 6, 3],
            filter_sizes=[
                [64, 64, 256],
                [128, 128, 512],
                [256, 256, 1024],
                [512, 512, 2048],
            ],
        ),
    ),
    LayerArchitecture(
        layers=101,
        size=ModelSize.large,
        filter_spec=FilterSpecification(
            repetitions=[3, 4, 23, 3],
            filter_sizes=[
                [64, 64, 256],
                [128, 128, 512],
                [256, 256, 1024],
                [512, 512, 2048],
            ],
        ),
    ),
    LayerArchitecture(
        layers=152,
        size=ModelSize.large,
        filter_spec=FilterSpecification(
            repetitions=[3, 8, 36, 3],
            filter_sizes=[
                [64, 64, 256],
                [128, 128, 512],
                [256, 256, 1024],
                [512, 512, 2048],
            ],
        ),
    ),
]


def conv_x(x: int, architecture: LayerArchitecture) -> Model:
    layers_func = (
        resnet_large_conv_layer if architecture.size == ModelSize.large else resnet_small_conv_layer
    )
    layer_repetitions = architecture.filter_spec.repetitions[x]
    layer_filters = architecture.filter_spec.filter_sizes[x]

    layers_input = Input(shape=(56, 56))
    layers = [
        *multi_layers(layers_func, n=layer_repetitions - 1, filters=layer_filters),
        *layers_func(filters=layer_filters, final_batch=False),
    ]
    layers = call_reduce(layers, layer_input)
    residual = Add()([layers_input, layers])
    return Model(inputs=layers_input, outputs=residual, name=f"conv{x+2}")
