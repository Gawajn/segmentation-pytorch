import argparse
import json
import re
from dataclasses import dataclass, field
from typing import Tuple, List

from mashumaro.mixins.json import DataClassJSONMixin

from segmentation.modules import Architecture
from segmentation.optimizer import Optimizers
from segmentation.settings import ModelFile, ModelConfiguration, ProcessingSettings, CustomModelSettings, \
    Preprocessingfunction, PredefinedNetworkSettings, ColorMap, ClassSpec


@dataclass
class HistoricalCustomModelSettings(DataClassJSONMixin):
    CLASSES: int
    ENCODER_FILTER: List[int]
    DECODER_FILTER: List[int]
    ATTENTION_ENCODER_FILTER: List[int]
    TYPE: str = "attentionunet"
    KERNEL_SIZE: int = 3
    PADDING: int = 1
    STRIDE: int = 1
    ENCODER_DEPTH: int = 3
    ATTENTION_DEPTH: int = 3
    ATTENTION_ENCODER_DEPTH: int = 3
    ACTIVATION: bool = False
    CHANNELS_IN: int = 3
    CHANNELS_OUT: int = 16
    ATTENTION: bool = True

    def get_kwargs(self):
        return {
            "in_channels": self.CHANNELS_IN,
            "out_channels": self.CHANNELS_OUT,
            "n_class": self.CLASSES,
            "kernel_size": self.KERNEL_SIZE,
            "padding": self.PADDING,
            "stride": self.STRIDE,
            "attention": self.ATTENTION,
            "encoder_depth": self.ENCODER_DEPTH,
            "attention_depth": self.ATTENTION_DEPTH,
            "encoder_filter": self.ENCODER_FILTER,
            "decoder_filter": self.DECODER_FILTER,
            "attention_encoder_filter": self.ATTENTION_ENCODER_FILTER,
            "attention_encoder_depth": self.ATTENTION_ENCODER_DEPTH,
        }
    def to_custom_model_settings(self, weight_sharing = False, scaled_image_input=False):

        return CustomModelSettings(classes=self.CLASSES, encoder_filter=self.ENCODER_FILTER, decoder_filter=self.DECODER_FILTER, attention_encoder_filter=self.ATTENTION_ENCODER_FILTER, type=self.TYPE, kernel_size=self.KERNEL_SIZE, padding=self.PADDING, stride=self.STRIDE
                                   , encoder_depth=self.ENCODER_DEPTH, attention_depth=self.ATTENTION_DEPTH, attention_encoder_depth=self.ATTENTION_ENCODER_DEPTH, activation=self.ACTIVATION, channels_in=self.CHANNELS_IN, channels_out=self.CHANNELS_OUT, attention=self.ATTENTION,
                                   weight_sharing=weight_sharing, scaled_image_input=scaled_image_input)

@dataclass
class HistoricalTrainSettings(DataClassJSONMixin):
    CLASSES: int
    OUTPUT_PATH: str

    EPOCHS: int = 15
    OPTIMIZER: Optimizers = Optimizers.ADAM
    LEARNINGRATE_ENCODER: float = 1.e-5
    LEARNINGRATE_DECODER: float = 1.e-4
    LEARNINGRATE_SEGHEAD: float = 1.e-4
    PADDING_VALUE: int = 32

    CUSTOM_MODEL: HistoricalCustomModelSettings = None
    DECODER_CHANNELS: Tuple[int, ...] = field(default_factory=tuple)
    ENCODER_DEPTH: int = 5
    ENCODER: str = 'efficientnet-b3'
    BATCH_ACCUMULATION: int = 8
    TRAIN_BATCH_SIZE: int = 1
    VAL_BATCH_SIZE: int = 1
    ARCHITECTURE: Architecture = Architecture.UNET
    MODEL_PATH: str = None
    IMAGEMAX_AREA: int = 1000000

    PROCESSES: int = 0

    def get_processing_settings(self, rgb:bool):
        return ProcessingSettings(input_padding_value=self.PADDING_VALUE, rgb=rgb,
                                  preprocessing=Preprocessingfunction(self.ENCODER if not self.CUSTOM_MODEL else "default"),scale_train=True, scale_predict=True, scale_max_area=self.IMAGEMAX_AREA)

    def get_network_settings(self):
        return PredefinedNetworkSettings(self.CLASSES,self.ARCHITECTURE,self.ENCODER,self.ENCODER_DEPTH,self.DECODER_CHANNELS)


def main():
    from segmentation.network import Network
    from segmentation.settings import Architecture
    from segmentation.modules import ENCODERS

    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, default=None,
                        help="load a json in the old file format and parse it")
    parser.add_argument("--color_map", type=str, required=False,
                        help="path to color map to load")
    parser.add_argument("-o", type=str, help="name of the output file (.json)", required=False)

    args = parser.parse_args()

    with open(args.json) as f:
        settings = HistoricalTrainSettings.from_json(f.read())

    color_map = None
    if args.color_map:
        with open(args.color_map) as f:
            cmd = json.loads(f.read())
            cm = ColorMap([])
            for color, ll in cmd.items():
                color: str = color
                color = color.strip()[1:-1]
                colors = list(map(int, color.split(",")))
                label = ll[0]
                name = ll[1]
                cm.class_spec.append(ClassSpec(int(label), str(name), colors))
        color_map = cm

    mf = ModelFile(ModelConfiguration(custom_model_settings=settings.CUSTOM_MODEL.to_custom_model_settings(scaled_image_input=False, weight_sharing=False) if settings.CUSTOM_MODEL else None,
                                      network_settings=settings.get_network_settings(),
                                      preprocessing_settings=settings.get_processing_settings(rgb=True),
                                      use_custom_model=True if settings.CUSTOM_MODEL else False,
                                      color_map=color_map),
                   None)
    if not args.o:
        print(mf)
    else:
        with open(args.o,"w") as of:
            of.write(json.dumps(mf.to_dict(), indent=4))

if __name__ == "__main__":
    main()

