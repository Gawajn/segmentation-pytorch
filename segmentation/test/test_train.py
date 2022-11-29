from pathlib import Path

from pagexml_mask_converter.pagexml_to_mask import MaskGenerator, PCGTSVersion, MaskSetting, MaskType
from torch.utils.data import DataLoader

from segmentation.callback import ModelWriterCallback
from segmentation.dataset import dirs_to_pandaframe, load_image_map_from_file, MaskDataset, compose
from segmentation.model_builder import ModelBuilderMeta
from segmentation.network import NetworkTrainer
from segmentation.settings import ModelConfiguration, CustomModelSettings, ProcessingSettings, NetworkTrainSettings, \
    ColorMap, ClassSpec

if __name__ == '__main__':
    a = dirs_to_pandaframe(
        ['/home/alexanderh/Documents/datasets/baselines/train/image/'],
        ['/home/alexanderh/Documents/datasets/baselines/train/page/'])

    b = dirs_to_pandaframe(
        ['/home/alexanderh/Documents/datasets/baselines/test/image/'],
        ['/home/alexanderh/Documents/datasets/baselines/test/page/'])

    cmap = ColorMap([ClassSpec(label=0, name="Background", color=[255, 255, 255]),
                     ClassSpec(label=1, name="Baseline", color=[255, 0, 255]),
                     ClassSpec(label=2, name="BaselineBorder", color=[255, 255, 0])])

    from segmentation.dataset import default_transform, dirs_to_pandaframe, load_image_map_from_file, XMLDataset

    settings = MaskSetting(MASK_TYPE=MaskType.BASE_LINE, PCGTS_VERSION=PCGTSVersion.PCGTS2013, LINEWIDTH=5,
                           BASELINELENGTH=10)
    dt = XMLDataset(a, cmap, transform=compose([default_transform()]),
                    mask_generator=MaskGenerator(settings=settings))
    d_test = XMLDataset(b, cmap, transform=compose([default_transform()]),
                        mask_generator=MaskGenerator(settings=settings))

    train_loader = DataLoader(dataset=dt, batch_size=1)
    val_loader = DataLoader(dataset=d_test, batch_size=1)



    config = ModelConfiguration(use_custom_model=True,
                                network_settings=None,
                                custom_model_settings=CustomModelSettings(classes=len(cmap),
                                                                          encoder_depth=4,
                                                                          encoder_filter=[16, 32, 64, 128, 256],
                                                                          decoder_filter=[16, 32, 64, 128, 256],
                                                                          attention_encoder_filter=[16, 32, 64, 128],
                                                                          ),
                                preprocessing_settings=ProcessingSettings(input_padding_value=64,rgb=True),
                                color_map=cmap)


    network = ModelBuilderMeta(config, "cuda").get_model()

    mw = ModelWriterCallback(network, config, save_path=Path("/tmp"))
    trainer = NetworkTrainer(network, NetworkTrainSettings(), "cuda",
                             callbacks=[mw], debug_color_map=config.color_map)

    trainer.train_epochs(train_loader=train_loader, val_loader=val_loader, n_epoch=2, lr_schedule=None)

