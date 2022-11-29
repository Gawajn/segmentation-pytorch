import os
from pathlib import Path

import numpy as np
import torch.cuda

from segmentation.callback import ModelWriterCallback
from segmentation.model_builder import ModelBuilderMeta, ModelBuilderLoad
from segmentation.network import NetworkTrainer
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
        return torch.cuda.current_device()
    else:
        return "cpu"


def parse_arguments():
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
        if args.mode == "xml_region" or args.mode == "xml_baseline":
            dt = XMLDataset(train, color_map, transform=compose([default_transform()]),
                            mask_generator=MaskGenerator(settings=mask_settings))
            d_test = XMLDataset(test, color_map, transform=compose([default_transform()]),
                                mask_generator=MaskGenerator(settings=mask_settings))

        else:
            dt = MaskDataset(train, color_map, transform=compose([default_transform()]),
                             scale_area=args.scale_area,
                             crop=args.crop_train, crop_x=args.crop_x_train, crop_y=args.crop_y_train)
            d_test = MaskDataset(test, color_map, transform=compose([default_transform()]),
                                 scale_area=args.scale_area,
                                 crop=args.crop_val, crop_x=args.crop_x_val, crop_y=args.crop_y_val)

        train_loader = DataLoader(dataset=dt, batch_size=1)
        val_loader = DataLoader(dataset=d_test, batch_size=1)

        config = ModelConfiguration(use_custom_model=args.custom_model,
                                    network_settings=get_predefinedNetworkSettings(
                                        args) if not args.custom_model else None,
                                    custom_model_settings=get_custom_model_settings(
                                        args) if args.custom_model else None,
                                    preprocessing_settings=ProcessingSettings(input_padding_value=args.padding_value,
                                                                              rgb=True,
                                                                              scale_max_area=args.scale_area,
                                                                              preprocessing=Preprocessingfunction(
                                                                                  args.predefined_encoder if not args.custom_model else Preprocessingfunction.name)),
                                    color_map=color_map)

        network = ModelBuilderMeta(config, args.device).get_model()

        mw = ModelWriterCallback(network, config, save_path=Path(args.output_path), prefix=model_prefix)
        trainer = NetworkTrainer(network, NetworkTrainSettings(optimizer=Optimizers(args.optimizer),
                                                               learningrate_seghead=args.learning_rate,
                                                               batch_accumulation=args.batch_accumulation,
                                                               processes=args.processes,

                                                               ), args.device,
                                 callbacks=[mw], debug_color_map=config.color_map)

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
            total_accuracy = float(np.mean([mw.highest_accuracy for mw in model_writers]))
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
                accuracy, loss = test_network(ml.model, eval_loader, torch.nn.CrossEntropyLoss(), args.padding_value)
                total_accuracy += accuracy
                total_loss += loss
        print("EXPERIMENT_OUT=" + str(total_accuracy / len(model_writers)) + "," + str(total_loss / len(model_writers)))


if __name__ == "__main__":
    main()
