import argparse
import json
from dataclasses import dataclass, field
from typing import Tuple, List

import albumentations
from albumentations.pytorch import ToTensorV2
from mashumaro.mixins.json import DataClassJSONMixin

from segmentation.modules import Architecture
from segmentation.optimizer import Optimizers
from segmentation.preprocessing.workflow import PreprocessingTransforms, ColorMapTransform, GrayToRGBTransform, \
    NetworkEncoderTransform
from segmentation.settings import ModelFile, ModelConfiguration, ProcessingSettings, CustomModelSettings, \
    Preprocessingfunction, PredefinedNetworkSettings, ColorMap, ClassSpec
import sys


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

    def to_custom_model_settings(self, weight_sharing=False, scaled_image_input=False):
        return CustomModelSettings(classes=self.CLASSES, encoder_filter=self.ENCODER_FILTER,
                                   decoder_filter=self.DECODER_FILTER,
                                   attention_encoder_filter=self.ATTENTION_ENCODER_FILTER, type=self.TYPE,
                                   kernel_size=self.KERNEL_SIZE, padding=self.PADDING, stride=self.STRIDE
                                   , encoder_depth=self.ENCODER_DEPTH, attention_depth=self.ATTENTION_DEPTH,
                                   attention_encoder_depth=self.ATTENTION_ENCODER_DEPTH, activation=self.ACTIVATION,
                                   channels_in=self.CHANNELS_IN, channels_out=self.CHANNELS_OUT,
                                   attention=self.ATTENTION,
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

    def get_processing_settings(self, rgb: bool, color_map: ColorMap):
        def remove_nones(x):
            return [y for y in x if y is not None]

        def default_transform():
            result = albumentations.Compose([
                albumentations.HorizontalFlip(),
                albumentations.RandomGamma(),
                albumentations.RandomBrightnessContrast(),
                albumentations.OneOf([
                    albumentations.ToGray(),
                    albumentations.CLAHE()]),
                albumentations.RandomScale(),

            ])
            return result

        input_transforms = albumentations.Compose(remove_nones([
            GrayToRGBTransform() if rgb else None,
            ColorMapTransform(color_map=color_map.to_albumentation_color_map())

        ]))
        aug_transforms = default_transform()
        tta_transforms = None
        post_transforms = albumentations.Compose(remove_nones([
            NetworkEncoderTransform(self.ENCODER if not self.CUSTOM_MODEL else Preprocessingfunction.name),
            ToTensorV2()
        ]))
        transforms = PreprocessingTransforms(
            input_transform=input_transforms,
            aug_transform=aug_transforms,
            # tta_transforms=tta_transforms,
            post_transforms=post_transforms,
        )
        return ProcessingSettings(input_padding_value=self.PADDING_VALUE, rgb=rgb,
                                  preprocessing=Preprocessingfunction(
                                      self.ENCODER if not self.CUSTOM_MODEL else "default"), scale_train=True,
                                  scale_predict=True, scale_max_area=self.IMAGEMAX_AREA, transforms=transforms.to_dict())

    def get_network_settings(self):
        return PredefinedNetworkSettings(self.CLASSES, self.ARCHITECTURE, self.ENCODER, self.ENCODER_DEPTH,
                                         self.DECODER_CHANNELS)


def check_args(args):
    if bool(args.color_map is not None) == bool(args.use_predefined_baseline_color_map is not None):
        raise RuntimeError("Must provide either --color_map or --use_predefined_baseline_color_map")

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, default=None, required=True,
                        help="load a json in the old file format and parse it")
    parser.add_argument("--color_map", type=str, required=False,
                        help="path to color map to load")
    parser.add_argument("--use_predefined_baseline_color_map", required=False,
                        help="path to color map to load", action="store_true")
    parser.add_argument("-o", type=str, help="name of the output file (.json)", required=False)

    args = parser.parse_args()

    check_args(args)

    with open(args.json) as f:
        loaded_json_str = f.read()
        print(json.loads(loaded_json_str),file=sys.stderr)
        settings = HistoricalTrainSettings.from_json(loaded_json_str)

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

    if args.use_predefined_baseline_color_map:
        color_map = ColorMap([ClassSpec(label=0, name="Background", color=[255, 255, 255]),
                              ClassSpec(label=1, name="Baseline", color=[255, 0, 0]),
                              ClassSpec(label=2, name="BaselineBorder", color=[0, 255, 0])])

    mf = ModelFile(ModelConfiguration(
        custom_model_settings=settings.CUSTOM_MODEL.to_custom_model_settings(scaled_image_input=False,
                                                                             weight_sharing=False) if settings.CUSTOM_MODEL else None,
        network_settings=settings.get_network_settings(),
        preprocessing_settings=settings.get_processing_settings(rgb=True, color_map=color_map),
        use_custom_model=True if settings.CUSTOM_MODEL else False,
        color_map=color_map),
                   None)
    if not args.o:
        print(mf)
    else:
        with open(args.o, "w") as of:
            of.write(json.dumps(mf.to_dict(), indent=4))


if __name__ == "__main__":
    main()
