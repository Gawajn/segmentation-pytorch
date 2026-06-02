from dataclasses import dataclass, field
from typing import List, Any

import numpy as np
from mashumaro.mixins.json import DataClassJSONMixin
from mashumaro.types import SerializationStrategy

class TensorToFloatStrategy(SerializationStrategy):
    def serialize(self, value: Any) -> float:
        return float(value.item()) if hasattr(value, "item") else float(value)

    def deserialize(self, value: Any) -> Any:
        return value
@dataclass
class MetricStats:
    name: str = "none"
    values: List[float] = field(
        default_factory=list,
        metadata={"mashumaro": {"serialization_strategy": TensorToFloatStrategy()}}
    )

    def value(self):
        if not self.values:
            return 0.0
        cleaned = [v.item() if hasattr(v, "item") else v for v in self.values]
        return float(np.mean(cleaned))


@dataclass
class EpochStats(DataClassJSONMixin):
    stats: List[MetricStats]

    def __iter__(self):
        return iter(self.stats)

    def __len__(self):
        return len(self.stats)

    def to_flat_dict(self ) -> dict:
        return {i.name: float(i.value()) for i in self.stats}

    def __getitem__(self, item):
        return self.stats[item]
