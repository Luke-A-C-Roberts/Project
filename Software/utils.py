from keras import Input
from keras.layers import Layer

from itertools import chain, repeat
from typing import Any, List, Iterable, Callable
from copy import deepcopy
from functools import reduce


def multi_layers(
    layers_func: Callable[[Any, ...], list[Layer]], n: int, *args: Any, **kwargs: Any
) -> Iterable[Layer]:
    return chain.from_iterable(map(deepcopy, repeat(layers_func(*args, **kwargs), n)))


def pipeline(layers: list[Layer], layer_input: Input) -> Layer:
    return reduce(lambda v, f: f(v), layers, layer_input)


def pad(num: int) -> tuple[tuple[int, int], tuple[int, int]]:
    return ((num, num), (num, num))
