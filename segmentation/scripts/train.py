import os
from pathlib import Path

import albumentations
import numpy as np
import torch.cuda
from albumentations.pytorch import ToTensorV2
import albumentations as albu

from segmentation.callback import ModelWriterCallback, EarlyStoppingCallback
from segmentation.losses import Losses
from segmentation.metrics import Metrics, MetricReduction
from segmentation.model_builder import ModelBuilderMeta, ModelBuilderLoad
from segmentation.network import NetworkTrainer
from segmentation.preprocessing.workflow import PreprocessingTransforms, GrayToRGBTransform, ColorMapTransform, \
    NetworkEncoderTransform
from segmentation.settings import Architecture, NetworkTrainSettings, Preprocessingfunction, ColorMap, ClassSpec
from segmentation.modules import ENCODERS
import argparse
from os import path
import warnings

import loguru
from pagexml_mask_converter.pagexml_to_mask import MaskSetting, PCGTSVersion, MaskType, MaskGenerator
from torch.utils.data import DataLoader

from segmentation.dataset import dirs_to_pandaframe, XMLDataset, compose, MaskDataset
from segmentation.optimizer import Optimizers
from sklearn.model_selection import KFold
from segmentation.settings import color_map_load_helper, ModelConfiguration, ProcessingSettings, \
    PredefinedNetworkSettings, CustomModelSettings
from dataclasses import fields

warnings.simplefilter(action='ignore', category=FutureWarning)


def dir_path(string):
    p = Path(string)
    if not p.is_dir():
        raise NotADirectoryError(p)
    return p


def get_default(cls, field_name: str):
    for x in fields(cls):
        if x.name == field_name:
            return x.default
    raise RuntimeError("incorrect dataclass field name")


def get_default_device():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def parse_arguments():
    # print(dir(NetworkTrainSettings))
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", type=str, default=get_default_device())
    parser.add_argument("-L", "--learning_rate", type=float, default=NetworkTrainSettings.learningrate_seghead,
                        help="set learning rate")
    parser.add_argument("-o", "--output_path", type=dir_path, default=Path(os.getcwd()),
                        help="target directory for model and logs")
    parser.add_argument("--load", type=str, default=None,
                        help="load an existing model and continue training")
    parser.add_argument("-E", "--n_epoch", type=int, default=15,
                        help="number of epochs")
    parser.add_argument("--data-augmentation", action="store_true",
                        help="Enable data augmentation")

    parser.add_argument("--train_input", type=dir_path, nargs="+", default=[],
                        help="Path to folder(s) containing train images")
    parser.add_argument("--train_mask", type=dir_path, nargs="+", default=[],
                        help="Path to folder(s) containing train xmls/mask_png")

    parser.add_argument("--test_input", type=dir_path, nargs="*", default=[],
                        help="Path to folder(s) containing test images")
    parser.add_argument("--test_mask", type=dir_path, nargs="*", default=[],
                        help="Path to folder(s) containing test xmls/mask_png")

    parser.add_argument("--eval_input", type=dir_path, nargs="*", default=[],
                        help="Path to folder(s) containing eval images")
    parser.add_argument("--eval_mask", type=dir_path, nargs="*", default=[],
                        help="Path to folder(s) containing eval xmls/mask_png")

    parser.add_argument("--color_map", type=str, required=False,
                        help="path to color map to load")
    parser.add_argument("--mode", choices=["xml_baseline", "xml_region", "mask"], required=True)

    # Generic
    parser.add_argument('--batch_accumulation', default=NetworkTrainSettings.batch_accumulation, type=int)
    parser.add_argument('--processes', default=NetworkTrainSettings.processes, type=int)
    parser.add_argument('--folds', default=1, type=int)
    parser.add_argument('--eval', action="store_true", help="Starts evaluation on test set after training")
    parser.add_argument("--scale_area", type=int, default=ProcessingSettings.scale_max_area,
                        help="max pixel amount of an image")
    parser.add_argument("--padding_value", type=int, help="padding size of the image",
                        default=ProcessingSettings.input_padding_value)
    parser.add_argument('--optimizer', default=NetworkTrainSettings.optimizer.value, type=str,
                        choices=[x.value for x in list(Optimizers)])
    #metric settings
    parser.add_argument('--metrics', default=NetworkTrainSettings.default_metric(), type=str,
                        choices=[x.value for x in list(Metrics)], nargs='+')
    parser.add_argument('--metrics_watcher_index', default=0, type=int,
                        help="Index of metric used for comparing models, default=0")
    parser.add_argument('--metrics_reduction', default=NetworkTrainSettings.metric_reduction.value, type=str,
                        choices=[x.value for x in list(MetricReduction)],
                        help="Metric reduction")
    parser.add_argument('--metrics_weights', default=NetworkTrainSettings.class_weights, type=float, nargs="+",
                        help="Metric class weight, default=None (Only used when using weighted reduction")

    parser.add_argument('--loss', default=NetworkTrainSettings.loss.value, type=str,
                        choices=[x.value for x in list(Losses)])
    parser.add_argument('--early_stopping', type=int, default=0,help="Number of epochs after which training is stopped model didn't improve")
    # Predefined
    parser.add_argument('--predefined_architecture',
                        default=get_default(PredefinedNetworkSettings, "architecture"),
                        choices=[x.value for x in list(Architecture)],
                        type=str,
                        help='Network architecture to use for training')
    parser.add_argument('--predefined_encoder',
                        default=get_default(PredefinedNetworkSettings, "encoder"),
                        choices=ENCODERS,
                        type=str,
                        help='Network architecture to use for training')
    parser.add_argument('--predefined_encoder_depth',
                        default=get_default(PredefinedNetworkSettings, "encoder_depth"),
                        choices=[3, 4, 5],
                        type=int,
                        help='Network architecture depth to use for training')
    parser.add_argument('--predefined_decoder_channel',
                        nargs='+', type=int, default=get_default(PredefinedNetworkSettings, "decoder_channel"),
                        help='List of integers which specify **in_channels** parameter for convolutions used in decoder. Lenght of the list should be the same as **encoder_depth**')
    # Custom
    parser.add_argument('--custom_model', action="store_true",
                        help='Use Custom model for training')
    parser.add_argument("--custom_model_kernel_size", type=int, default=CustomModelSettings.kernel_size,
                        help="kernel size of the custom model")
    parser.add_argument("--custom_model_padding", type=int, default=get_default(CustomModelSettings, "padding"),
                        help="padding of the custom model")
    parser.add_argument("--custom_model_stride", type=int, default=get_default(CustomModelSettings, "stride"),
                        help="stride of the custom model")
    parser.add_argument("--custom_model_encoder_depth", type=int,
                        default=get_default(CustomModelSettings, "encoder_depth"),
                        help="encoder depth of the custom model")
    parser.add_argument("--custom_model_attention_encoder_depth", type=int,
                        default=get_default(CustomModelSettings, "attention_encoder_depth"),
                        help="attention_encoder depth of the custom model")
    parser.add_argument("--custom_model_use_attention", action="store_true", help="use attention for the custom model")
    parser.add_argument("--custom_model_attention_depth", type=int,
                        default=get_default(CustomModelSettings, "attention_depth"),
                        help="attention depth of the custom model")
    parser.add_argument('--custom_model_encoder_filter', nargs='+', type=int,
                        help="filter of the encoder of the custom model. Number of filters should be equal to enocder depth + 1")
    parser.add_argument('--custom_model_decoder_filter', nargs='+', type=int,
                        help="filter of the decoder of the custom model. Number of filters should be equal to encoder depth + 1")
    parser.add_argument('--custom_model_encoder_attention_filter', nargs='+', type=int,
                        help="filter of the attention encoder of the custom model. Number of filters should be equal to attention depth + 1")
    parser.add_argument("--custom_model_no_weight_sharing", action="store_false",
                        help="weight sharing of models for scaled images")
    parser.add_argument("--custom_model_scaled_image_input", action="store_true", help="scaled image input")

    # Special Train Configuration Settings

    parser.add_argument('--batch_size_train', type=int, default=1,
                        help='batch_size')
    parser.add_argument('--batch_size_val', type=int, default=1,
                        help='batch_size')
    parser.add_argument('--crop_train', action="store_true",
                        help='crops train images')
    parser.add_argument('--crop_val', action="store_true",
                        help='crops train images')
    parser.add_argument('--crop_x_train', type=int, default=512,
                        help='crops train images')
    parser.add_argument('--crop_y_train', type=int, default=512,
                        help='crops train images')
    parser.add_argument('--crop_x_val', type=int, default=512,
                        help='crops train images')
    parser.add_argument('--crop_y_val', type=int, default=512,
                        help='crops train images')
    parser.add_argument('--seed', default=123, type=int)
    return parser.parse_args()


def main():
    args = parse_arguments()
    train = dirs_to_pandaframe(args.train_input, args.train_mask)[:100]
    test = dirs_to_pandaframe(args.test_input, args.test_mask)[:5]
    device = args.device

    if len(test) == 0:
        test = train
        loguru.logger.warning("Using train dataset as test, because test is empty")

    if args.mode == "mask":
        color_map = color_map_load_helper(args.color_map)
    elif args.mode == "xml_baseline":
        color_map = ColorMap([ClassSpec(label=0, name="Background", color=[255, 255, 255]),
                              ClassSpec(label=1, name="Baseline", color=[255, 0, 0]),
                              ClassSpec(label=2, name="BaselineBorder", color=[0, 255, 0])])
    else:
        raise NotImplementedError()

    from segmentation.dataset import default_transform
    mask_settings = MaskSetting(MASK_TYPE=MaskType.BASE_LINE if args.mode == "xml_baseline" else MaskType.ALLTYPES,
                                PCGTS_VERSION=PCGTSVersion.PCGTS2013, LINEWIDTH=5,
                                BASELINELENGTH=10)

    def get_predefinedNetworkSettings(args):
        return PredefinedNetworkSettings(architecture=Architecture(args.predefined_architecture),
                                         encoder=args.predefined_encoder,
                                         classes=len(color_map),
                                         encoder_depth=args.predefined_encoder_depth,
                                         decoder_channel=args.predefined_decoder_channel)

    def get_custom_model_settings(args) -> CustomModelSettings:
        return CustomModelSettings(
            encoder_filter=args.custom_model_encoder_filter,
            decoder_filter=args.custom_model_decoder_filter,
            attention_encoder_filter=args.custom_model_encoder_attention_filter,
            attention=args.custom_model_use_attention,
            classes=len(color_map),
            attention_depth=args.custom_model_attention_depth,
            encoder_depth=args.custom_model_encoder_depth,
            attention_encoder_depth=args.custom_model_attention_encoder_depth,
            stride=args.custom_model_stride,
            padding=args.custom_model_padding,
            kernel_size=args.custom_model_kernel_size,
            weight_sharing=False if args.custom_model_no_weight_sharing else True,
            scaled_image_input=args.custom_model_scaled_image_input,
        )

    def train_arg(train, test, args, model_prefix="") -> ModelWriterCallback:
        """
        def process(image, mask, rgb, preprocessing, apply_preprocessing, augmentation, color_map=None,
            binary_augmentation=True, ocropy=True, crop=False, crop_x=512, crop_y=512):
    if rgb:
        image = gray_to_rgb(image)
    result = {"image": image}
    if color_map:
        if mask.ndim == 3:
            result["mask"] = color_to_label(mask, color_map)
        elif mask.ndim == 2:
            u_values = np.unique(mask)
            mask = result["mask"]
            for ind, x in enumerate(u_values):
                mask[mask == x] = ind
            result["mask"] = mask
    else:
        result["mask"] = mask if mask is not None else image

    if augmentation is not None:
        result = augmentation(**result)

    if augmentation is not None and binary_augmentation:
        from segmentation.preprocessing.basic_binarizer import gauss_threshold
        from segmentation.preprocessing.ocrupus import binarize
        ran = np.random.randint(1, 5)
        if ran == 1:
            if ocropy:
                binary = binarize(result["image"].astype("float64")).astype("uint8") * 255
                gray = gray_to_rgb(binary)
                result["image"] = gray_to_rgb(gray)
        if ran == 2:
            image = rgb2gray(result["image"]).astype(np.uint8)
            result["image"] = gray_to_rgb(gauss_threshold(image))
    if crop:
        result = compose([[albu.RandomCrop(
            crop_y, crop_x, p=1
        )]])(**result)
        ## albumentations.augmentations.crops.transforms.RandomResizedCrop
    if apply_preprocessing is not None and apply_preprocessing:
        result["image"] = preprocessing(result["image"])
    result = compose([post_transforms()])(**result)
    return result["image"], result["mask"]

        """
        def remove_nones(x):
            return [y for y in x if y is not None]

        def default_transform():
            result = albumentations.Compose([
                albu.HorizontalFlip(),
                albu.RandomGamma(),
                albu.RandomBrightnessContrast(),
                albu.OneOf([
                    albu.ToGray(),
                    albu.CLAHE()]),
                albu.RandomScale(),

            ])
            return result
        input_transforms = albumentations.Compose(remove_nones([
            GrayToRGBTransform() if True else None,
            ColorMapTransform(color_map=color_map.to_albumentation_color_map())

        ]))
        aug_transforms = default_transform()
        tta_transforms = None
        post_transforms = albumentations.Compose(remove_nones([
            NetworkEncoderTransform(args.predefined_encoder if not args.custom_model else Preprocessingfunction.name),
            ToTensorV2()
        ]))
        transforms = PreprocessingTransforms(
            input_transform=input_transforms,
            aug_transform=aug_transforms,
            #tta_transforms=tta_transforms,
            post_transforms=post_transforms,
        )

        if args.mode == "xml_region" or args.mode == "xml_baseline":
            dt = XMLDataset(train, transform=transforms,
                            mask_generator=MaskGenerator(settings=mask_settings))
            d_test = XMLDataset(test, transform=transforms,
                                mask_generator=MaskGenerator(settings=mask_settings))

        else:
            dt = MaskDataset(train, transform=transforms, scale_area=args.scale_area)
            d_test = MaskDataset(test, transform=transforms, scale_area=args.scale_area)

        train_loader = DataLoader(dataset=dt, batch_size=1)
        val_loader = DataLoader(dataset=d_test, batch_size=1)

        config = ModelConfiguration(use_custom_model=args.custom_model,
                                    network_settings=get_predefinedNetworkSettings(
                                        args) if not args.custom_model else None,
                                    custom_model_settings=get_custom_model_settings(
                                        args) if args.custom_model else None,
                                    preprocessing_settings=ProcessingSettings(transforms=transforms.to_dict(),
                                                                              input_padding_value=args.padding_value,
                                                                              rgb=True,
                                                                              scale_max_area=args.scale_area,
                                                                              preprocessing=Preprocessingfunction(
                                                                                  args.predefined_encoder if not args.custom_model else Preprocessingfunction.name)),
                                    color_map=color_map)

        network = ModelBuilderMeta(config, args.device).get_model()
        mw = ModelWriterCallback(network, config, save_path=Path(args.output_path), prefix=model_prefix, metric_watcher_index=args.metrics_watcher_index)

        callbacks = [mw]
        if args.early_stopping > 0:
            me = EarlyStoppingCallback(patience=args.early_stopping, metric_watcher_index=args.metrics_watcher_index)
            callbacks.append(me)
        trainer = NetworkTrainer(network, NetworkTrainSettings(classes=len(color_map),
                                                               optimizer=Optimizers(args.optimizer),
                                                               learningrate_seghead=args.learning_rate,
                                                               batch_accumulation=args.batch_accumulation,
                                                               processes=args.processes,
                                                               metrics=[Metrics(x) for x in args.metrics],
                                                               metric_reduction=MetricReduction(args.metrics_reduction),
                                                               watcher_metric_index=args.metrics_watcher_index,
                                                               class_weights=args.metrics_weights,
                                                               loss=Losses(args.loss),
                                                               ), args.device,
                                 callbacks=callbacks, debug_color_map=config.color_map)

        trainer.train_epochs(train_loader=train_loader, val_loader=val_loader, n_epoch=args.n_epoch, lr_schedule=None)
        return mw

    model_writers = []
    if args.test_input == [] and args.folds > 1:
        kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        for ind, x in enumerate(kf.split(train, None)):
            train_f = train.iloc[x[0]].reset_index(drop=True)
            test_f = train.iloc[x[1]].reset_index(drop=True)
            model_writers.append(train_arg(train_f, test_f, args=args, model_prefix=f"fold{ind}"))

    else:
        model_writers.append(train_arg(train=train, test=test, args=args))

    if args.eval:
        total_accuracy = 0
        total_loss = 0
        if not args.eval_images:
            total_accuracy = float(np.mean([mw.stats[mw.metric_watcher_index].value() for mw in model_writers]))
            total_loss = float(np.mean([mw.best_loss for mw in model_writers]))

        elif args.eval_images:
            total_accuracy = 0
            total_loss = 0
            for ind, x in enumerate(model_writers):
                ml = ModelBuilderLoad.from_disk(x.get_best_model_path(), device=device).get_model()
                eval_df = dirs_to_pandaframe(args.eval_input, args.eval_mask)
                if args.mode == "xml_region" or args.mode == "xml_baseline":
                    d_eval = XMLDataset(eval_df, color_map, transform=compose([default_transform()]),
                                        mask_generator=MaskGenerator(settings=mask_settings))

                else:
                    d_eval = MaskDataset(eval_df, color_map, transform=compose([default_transform()]),
                                         scale_area=args.scale_area)
                eval_loader = DataLoader(dataset=d_eval, batch_size=1)

                from segmentation.network import test as test_network
                accuracy, loss = test_network(ml.model, device, eval_loader, Losses(args.loss).get_loss()() if Losses(args.loss) == Losses.cross_entropy_loss else Losses(args.loss).get_loss()(mode="multiclass"), classes=len(color_map),
                                              metrics=[Metrics(x) for x in args.metrics], metric_reduction=MetricReduction(args.metrics_reduction),  metric_watcher_index=args.metrics_watcher_index, class_weights=args.metrics_weights,
                                              padding_value=args.padding_value)
                total_accuracy += accuracy.stats[args.metrics_watcher_index].value()
                total_loss += loss
        print("EXPERIMENT_OUT=" + str(total_accuracy / len(model_writers)) + "," + str(total_loss / len(model_writers)))


if __name__ == "__main__":
    main()
