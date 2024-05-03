from keras.api import Input
from keras.api.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Layer,
    MaxPooling2D,
    ZeroPadding2D
)

from itertools import chain, repeat
from typing import Any, List, Iterator, Callable
from copy import deepcopy
from functools import reduce, partial


def compose(*funcs: Callable) -> Callable:
    """
    composes two or more functions together. (f•g)(...) = g(f(...))
    """
    return lambda *args, **kwargs: reduce(
        lambda v, f: f(v),
        funcs[:-1],
        funcs[-1](*args, **kwargs)
    )


def multi_layers(layers_func: partial[list[Layer]], n: int) -> Iterator[Layer]:
    """
    for when there needs to be many of the same layer, multi_layers creates
    an iterator.
    """
    return chain.from_iterable(map(deepcopy, repeat(layers_func(), n)))


def pipeline(layers: list[Layer], layer_input: Any) -> Layer:
    """
    repeatedly connects layers together
    """
    return reduce(lambda v, f: f(v), layers, layer_input)


def pad(num: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    creates a temporary tuple for padding containing 2×2 of the same number
    """
    return ((num, num), (num, num))






