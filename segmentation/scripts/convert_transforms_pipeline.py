import argparse
import json
import re
from pathlib import Path

import albumentations
from albumentations.pytorch import ToTensorV2
from segmentation.preprocessing.workflow import PreprocessingTransforms, ColorMapTransform, GrayToRGBTransform, \
    NetworkEncoderTransform
from segmentation.settings import ModelFile, ModelConfiguration, ProcessingSettings, CustomModelSettings, \
    Preprocessingfunction, PredefinedNetworkSettings, ColorMap, ClassSpec


def main():


    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, default=None,
                        help="load a json in the old file format and parse it")
    parser.add_argument("-o", type=str, help="name of the output file (.json)", required=False)

    args = parser.parse_args()

    settings = ModelFile.from_file(Path(args.json))
    def get_processing_settings(rgb: bool, color_map: ColorMap, preprocessingfunc):
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
            NetworkEncoderTransform(preprocessingfunc),
            ToTensorV2()
        ]))
        transforms = PreprocessingTransforms(
            input_transform=input_transforms,
            aug_transform=aug_transforms,
            # tta_transforms=tta_transforms,
            post_transforms=post_transforms,
        )
        return transforms
    preprocessingfunc = settings.model_configuration.preprocessing_settings.preprocessing.name

    settings.model_configuration.preprocessing_settings = ProcessingSettings(
        input_padding_value=settings.model_configuration.preprocessing_settings.input_padding_value,
        rgb=settings.model_configuration.preprocessing_settings.rgb,
        preprocessing=settings.model_configuration.preprocessing_settings.preprocessing,
        scale_train=settings.model_configuration.preprocessing_settings.scale_train,
        scale_predict=settings.model_configuration.preprocessing_settings.scale_predict,
        scale_max_area=settings.model_configuration.preprocessing_settings.scale_max_area,
        transforms=get_processing_settings(settings.model_configuration.preprocessing_settings.rgb, settings.model_configuration.color_map, preprocessingfunc).to_dict())

    if not args.o:
        print(settings)
    else:
        with open(args.o, "w") as of:
            of.write(json.dumps(settings.to_dict(), indent=4))


if __name__ == "__main__":
    main()
