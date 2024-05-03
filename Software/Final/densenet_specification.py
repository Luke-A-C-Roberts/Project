from model_specification import LayerSpec, MultiLayerSpec

from dataclasses import dataclass
from typing import Iterable, Iterator


@dataclass(frozen=True)
class DensenetSpec(Iterable):
    multi_layer_specs: list[MultiLayerSpec]

    def __iter__(self) -> Iterator:
        return iter(self.multi_layer_specs)

    def __len__(self) -> int:
        return len(self.multi_layer_specs)


DENSENET_SPECIFICATIONS: dict[int, DensenetSpec] = {
    121: DensenetSpec([
        MultiLayerSpec([LayerSpec(1, 128), LayerSpec(3, 32)], 6),
        MultiLayerSpec([LayerSpec(1, 128), LayerSpec(3, 32)], 12),
        MultiLayerSpec([LayerSpec(1, 128), LayerSpec(3, 32)], 24),
        MultiLayerSpec([LayerSpec(1, 128), LayerSpec(3, 32)], 16),
    ]),
    169: DensenetSpec([
        MultiLayerSpec([LayerSpec(1, 128), LayerSpec(3, 32)], 6),
        MultiLayerSpec([LayerSpec(1, 128), LayerSpec(3, 32)], 12),
        MultiLayerSpec([LayerSpec(1, 128), LayerSpec(3, 32)], 32),
        MultiLayerSpec([LayerSpec(1, 128), LayerSpec(3, 32)], 32),
    ]),
    201: DensenetSpec([
        MultiLayerSpec([LayerSpec(1, 128), LayerSpec(3, 32)], 6),
        MultiLayerSpec([LayerSpec(1, 128), LayerSpec(3, 32)], 12),
        MultiLayerSpec([LayerSpec(1, 128), LayerSpec(3, 32)], 48),
        MultiLayerSpec([LayerSpec(1, 128), LayerSpec(3, 32)], 32),
    ]),
    264: DensenetSpec([
        MultiLayerSpec([LayerSpec(1, 128), LayerSpec(3, 32)], 6),
        MultiLayerSpec([LayerSpec(1, 128), LayerSpec(3, 32)], 12),
        MultiLayerSpec([LayerSpec(1, 128), LayerSpec(3, 32)], 64),
        MultiLayerSpec([LayerSpec(1, 128), LayerSpec(3, 32)], 48),
    ])
}
