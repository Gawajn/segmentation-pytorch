from pathlib import Path

import albumentations
from albumentations.pytorch import ToTensorV2
from pagexml_mask_converter.pagexml_to_mask import MaskGenerator, PCGTSVersion, MaskSetting, MaskType
from torch.utils.data import DataLoader

from segmentation.callback import ModelWriterCallback
from segmentation.dataset import dirs_to_pandaframe, load_image_map_from_file, MaskDataset, compose
from segmentation.model_builder import ModelBuilderMeta
from segmentation.modules import Architecture
from segmentation.network import NetworkTrainer
from segmentation.preprocessing.workflow import PreprocessingTransforms, GrayToRGBTransform, ColorMapTransform, \
    NetworkEncoderTransform
from segmentation.settings import ModelConfiguration, CustomModelSettings, ProcessingSettings, NetworkTrainSettings, \
    ColorMap, ClassSpec, PredefinedNetworkSettings

if __name__ == '__main__':
    a = dirs_to_pandaframe(
        ['/home/alexanderh/Documents/datasets/baselines/train/image/'],
        ['/home/alexanderh/Documents/datasets/baselines/train/page/'])

    b = dirs_to_pandaframe(
        ['/home/alexanderh/Documents/datasets/baselines/test/image/'],
        ['/home/alexanderh/Documents/datasets/baselines/test/page/'])


    def remove_nones(x):
        return [y for y in x if y is not None]


    def default_transform():
        albu = albumentations
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


    cmap = ColorMap([ClassSpec(label=0, name="Background", color=[255, 255, 255]),
                     ClassSpec(label=1, name="Baseline", color=[255, 0, 0]),
                     ClassSpec(label=2, name="BaselineBorder", color=[0, 255, 0])])

    predef = PredefinedNetworkSettings(architecture=Architecture.UNET,
                                       classes=len(cmap))

    input_transforms = albumentations.Compose(remove_nones([
        GrayToRGBTransform() if True else None,
        ColorMapTransform(color_map=cmap.to_albumentation_color_map())
    ]))

    aug_transforms = default_transform()
    tta_transforms = None

    post_transforms = albumentations.Compose(remove_nones([
        NetworkEncoderTransform(predef.encoder),
        ToTensorV2()
    ]))

    transforms = PreprocessingTransforms(
        input_transform=input_transforms,
        aug_transform=aug_transforms,
        # tta_transforms=tta_transforms,
        post_transforms=post_transforms,
    )

    from segmentation.dataset import default_transform, dirs_to_pandaframe, load_image_map_from_file, XMLDataset

    settings = MaskSetting(MASK_TYPE=MaskType.BASE_LINE, PCGTS_VERSION=PCGTSVersion.PCGTS2013, LINEWIDTH=5,
                           BASELINELENGTH=10)
    dt = XMLDataset(a, transforms=transforms,
                    mask_generator=MaskGenerator(settings=settings))
    d_test = XMLDataset(a[:5], transforms=transforms.get_test_transforms(),
                        mask_generator=MaskGenerator(settings=settings))

    train_loader = DataLoader(dataset=dt, batch_size=1)
    val_loader = DataLoader(dataset=d_test, batch_size=1)

    config = ModelConfiguration(use_custom_model=False,
                                network_settings=predef,
                                custom_model_settings=CustomModelSettings(classes=len(cmap),
                                                                          encoder_depth=3,
                                                                          encoder_filter=[16, 32, 64, 128],
                                                                          decoder_filter=[16, 32, 64, 128],
                                                                          attention_encoder_filter=[16, 32, 64, 128],
                                                                          attention=False
                                                                          ),
                                preprocessing_settings=ProcessingSettings(transforms.to_dict(), input_padding_value=32,
                                                                          rgb=True),
                                color_map=cmap)

    network = ModelBuilderMeta(config, "cuda").get_model()

    mw = ModelWriterCallback(network, config, save_path=Path("/tmp"))
    trainer = NetworkTrainer(network, NetworkTrainSettings(classes=len(cmap)), "cuda",
                             callbacks=[mw], debug_color_map=config.color_map)

    trainer.train_epochs(train_loader=train_loader, val_loader=val_loader, n_epoch=1, lr_schedule=None)
