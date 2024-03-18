from keras import Input
from keras.layers import Layer

from itertools import chain, repeat
from typing import Any, List, Iterator, Callable
from copy import deepcopy
from functools import reduce, partial


def multi_layers(layers_func: partial[list[Layer]], n: int) -> Iterator[Layer]:
    return chain.from_iterable(map(deepcopy, repeat(layers_func(), n)))


def pipeline(layers: list[Layer], layer_input: Any) -> Layer:
    return reduce(lambda v, f: f(v), layers, layer_input)


def pad(num: int) -> tuple[tuple[int, int], tuple[int, int]]:
    return ((num, num), (num, num))
