from dataclasses import dataclass
from typing import Iterable, Iterator


@dataclass(frozen=True)
class LayerSpec(Iterable):
    """
    lowest level of architecture hierarchy. `filters` is used to repeat the
    layer many times
    """

    kernel_size: int|None
    filters: int

    def __iter__(self) -> Iterator:
        return iter([self.kernel_size, self.filters])


@dataclass(frozen=True)
class MultiLayerSpec(Iterable):
    """
    each section of the resnet model specifies which layers are present and how
    many times to repeat the layer. `use_strides` is used when the
    """

    layer_specs: list[LayerSpec]
    repetitions: int
    use_strides: bool = True

    def __iter__(self) -> Iterator:
        return iter(map(tuple, self.layer_specs))

    def __len__(self) -> int:
        return len(self.layer_specs)
