from model_specification import LayerSpec, MultiLayerSpec

from dataclasses import dataclass
from typing import Iterable, Iterator


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
