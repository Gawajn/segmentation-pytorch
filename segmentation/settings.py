from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Union, Tuple, Dict

import loguru
from dataclasses_json import dataclass_json

from segmentation.losses import Losses
from segmentation.metrics import MetricReduction, Metrics
from segmentation.modules import Architecture
import segmentation_models_pytorch as sm

from segmentation.optimizer import Optimizers
from mashumaro.mixins.json import DataClassJSONMixin


# from serde import serde


@dataclass
class CustomModelSettings(DataClassJSONMixin):
    classes: int
    encoder_filter: Union[None, List[int]]
    decoder_filter: Union[None, List[int]]
    attention_encoder_filter: Union[None, List[int]]
    type: str = "attentionunet"
    kernel_size: int = 3
    padding: int = 1
    stride: int = 1
    encoder_depth: int = 3
    attention_depth: int = 3
    attention_encoder_depth: int = 3
    activation: bool = True
    channels_in: int = 3
    channels_out: int = 16
    attention: bool = False
    weight_sharing: bool = True
    scaled_image_input: bool = False

    def get_kwargs(self):
        return {
            "in_channels": self.channels_in,
            "out_channels": self.channels_out,
            "n_class": self.classes,
            "kernel_size": self.kernel_size,
            "padding": self.padding,
            "stride": self.stride,
            "attention": self.attention,
            "encoder_depth": self.encoder_depth,
            "attention_depth": self.attention_depth,
            "encoder_filter": self.encoder_filter,
            "decoder_filter": self.decoder_filter,
            "attention_encoder_filter": self.attention_encoder_filter,
            "attention_encoder_depth": self.attention_encoder_depth,
            "weight_sharing": self.weight_sharing,
            "scaled_images_input": self.scaled_image_input,
        }


@dataclass
class NetworkTrainSettings:
    classes: int
    optimizer: Optimizers = Optimizers.ADAM
    learningrate_encoder: float = 1.e-5
    learningrate_decoder: float = 1.e-4
    learningrate_seghead: float = 1.e-4
    batch_accumulation: int = 1
    loss: Losses = Losses.cross_entropy_loss
    metric_reduction: MetricReduction = MetricReduction.micro
    metrics: List[Metrics] = field(default_factory=lambda: [NetworkTrainSettings.default_metric()])
    watcher_metric_index: int = 0
    class_weights: Union[None, List[float]] = None
    processes: int = 0
    ## additional heads
    additional_heads: int = 0
    additional_classes: List[int] = field(default_factory=list)

    @staticmethod
    def default_metric():
        return Metrics.accuracy


@dataclass
class PredefinedNetworkSettings(DataClassJSONMixin):
    classes: int
    architecture: Architecture = Architecture.UNET
    encoder: str = "efficientnet-b3"

    encoder_depth: int = 5
    decoder_channel: Tuple[int, ...] = (256, 128, 64, 32, 16)
    use_batch_norm_layer: bool = False

    add_number_of_heads: int = 1
    add_classes: List[int] = field(default_factory=list)

@dataclass
class ClassSpec(DataClassJSONMixin):
    label: int
    name: str
    color: Union[int, List[int]]


@dataclass
class ColorMap(DataClassJSONMixin):
    class_spec: List[ClassSpec]

    def __iter__(self):
        return iter(self.class_spec)

    def __len__(self):
        return len(self.class_spec)

    @classmethod
    def from_file(cls, path: Path) -> 'ColorMap':
        with open(path) as f:
            return cls.from_json(f.read())

    def to_albumentation_color_map(self) -> Dict[int,Tuple[int]]:
        return {x.label: x.color for x in self.class_spec}


@dataclass
class Preprocessingfunction(DataClassJSONMixin):
    name: str = "default"

    def get_preprocessing_function(self):
        from segmentation.datasets.dataset import default_preprocessing

        if self.name == "default":
            return default_preprocessing
        else:
            try:
                return sm.encoders.get_preprocessing_fn(self.name)
            except Exception as e:
                loguru.logger.critical("Preprocessing function does not exists")
                raise e


@dataclass
class ProcessingSettings(DataClassJSONMixin):
    transforms: Dict
    input_padding_value: int = 32
    rgb: bool = True
    preprocessing: Preprocessingfunction = field(
        default_factory=lambda: Preprocessingfunction(name="default"))
    scale_train: bool = True
    scale_predict: bool = True
    scale_max_area: Optional[int] = 1_000_000  # TODO implement



@dataclass
class ModelConfiguration(DataClassJSONMixin):
    use_custom_model: bool
    network_settings: Optional[PredefinedNetworkSettings]
    custom_model_settings: Optional[CustomModelSettings]
    color_map: ColorMap
    preprocessing_settings: ProcessingSettings


@dataclass
class ModelFile(DataClassJSONMixin):
    model_configuration: ModelConfiguration
    statistics: Optional[Dict[str, float]]

    @classmethod
    def from_file(cls, path: Path) -> 'ModelFile':
        with open(path) as f:
            return cls.from_json(f.read())


def color_map_load_helper(path: Path) -> ColorMap:
    try:
        res = ColorMap.from_file(path)
        return res
    except FileNotFoundError as e:
        loguru.logger.critical(f"No color map file present at {path}")
        raise e
    except Exception:
        pass

    try:
        res = ModelFile.from_file(path)
        return res.model_configuration.color_map
    except FileNotFoundError as e:
        loguru.logger.critical(f"No color map file present at {path}")
        raise e
    except Exception:
        pass

    raise RuntimeError("Cannot load model file. File doesn't exist or invalid format")
