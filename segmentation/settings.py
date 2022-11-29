from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Union, Tuple

import loguru
from dataclasses_json import dataclass_json

from segmentation.dataset import default_preprocessing
from segmentation.modules import Architecture
import segmentation_models_pytorch as sm

from segmentation.optimizer import Optimizers

@dataclass_json
@dataclass
class CustomModelSettings:
    classes: int
    encoder_filter: List[int]
    decoder_filter: List[int]
    attention_encoder_filter: List[int]
    type: str = "attentionunet"
    kernel_size: int = 3
    padding: int = 1
    stride: int = 1
    encoder_depth: int = 3
    attention_depth: int = 3
    attention_encoder_depth: int = 3
    activation: bool = False
    channels_in: int = 3
    channels_out: int = 16
    attention: bool = True
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


@dataclass_json
@dataclass
class NetworkTrainSettings:
    optimizer: Optimizers = Optimizers.ADAM
    learningrate_encoder: float = 1.e-5
    learningrate_decoder: float = 1.e-4
    learningrate_seghead: float = 1.e-4
    padding_value: int = 32

    batch_accumulation: int = 1

    processes: int = 0


@dataclass_json
@dataclass
class PredefinedNetworkSettings:
    architecture: Architecture = None
    encoder: str = None

    classes: int = None
    encoder_depth: int = None
    decoder_channel: Tuple[int, ...] = None


@dataclass_json
@dataclass
class ClassSpec:
    label: int
    name: str
    color: Union[int, List[int]]


@dataclass_json
@dataclass
class ColorMap:
    class_spec: List[ClassSpec]

    def __iter__(self):
        return iter(self.class_spec)

    def __len__(self):
        return len(self.class_spec)

    @classmethod
    def from_file(cls, path: Path) -> 'ColorMap':
        with open(path) as f:
            return cls.from_json(f.read())



@dataclass_json
@dataclass
class Preprocessingfunction:
    name: str

    def get_preprocessing_function(self):
        if self.name == "default":
            return default_preprocessing
        else:
            try:
                return sm.encoders.get_preprocessing_fn(self.name)
            except Exception as e:
                loguru.logger.critical("Preprocessing function does not exists")
                raise e


@dataclass_json
@dataclass
class ProcessingSettings:
    input_padding_value: int
    rgb: int = True
    preprocessing: Preprocessingfunction = field(
        default_factory=lambda: Preprocessingfunction(name="default"))
    scale_train: bool = True
    scale_predict: bool = True
    scale_max_area: Optional[int] = 1_000_000  # TODO implement


@dataclass_json
@dataclass
class ModelConfiguration:
    use_custom_model: bool
    network_settings: Optional[PredefinedNetworkSettings]
    custom_model_settings: Optional[CustomModelSettings]
    color_map: ColorMap
    preprocessing_settings: ProcessingSettings


@dataclass_json
@dataclass
class ModelFile:
    model_configuration: ModelConfiguration
    statistics: dict

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

