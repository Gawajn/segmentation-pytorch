"""End-to-end tests for multi-head (multi-task) training with optional per-head labels.

Covers:
 - training both model families (custom AttentionUnet and predefined smp UNet) with one
   additional head where half the samples lack the additional mask
 - save -> reload -> predict roundtrip incl. other_probability_map
 - regression: loading an old single-head checkpoint (scripts/best.torch) unchanged
 - warm-start: loading an old single-head checkpoint into a new multi-head model
"""
from pathlib import Path

import albumentations
import numpy as np
import pytest
import torch
from PIL import Image
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from segmentation.callback import ModelWriterCallback
from segmentation.datasets.dataset import dirs_to_pandaframe, MaskDataset
from segmentation.model_builder import ModelBuilderMeta, ModelBuilderLoad, load_weights_into
from segmentation.modules import Architecture
from segmentation.network import NetworkTrainer, NetworkPredictor
from segmentation.preprocessing.source_image import SourceImage
from segmentation.preprocessing.workflow import PreprocessingTransforms, GrayToRGBTransform, ColorMapTransform, \
    NetworkEncoderTransform
from segmentation.settings import ModelConfiguration, CustomModelSettings, PredefinedNetworkSettings, \
    ProcessingSettings, NetworkTrainSettings, ColorMap, ClassSpec, Preprocessingfunction, ModelFile

MAIN_CMAP = ColorMap([ClassSpec(label=0, name="Background", color=[255, 255, 255]),
                      ClassSpec(label=1, name="Baseline", color=[255, 0, 0]),
                      ClassSpec(label=2, name="BaselineBorder", color=[0, 255, 0])])
ADD_CMAP = ColorMap([ClassSpec(label=0, name="Background", color=[255, 255, 255]),
                     ClassSpec(label=1, name="Symbol", color=[0, 0, 255])])

OLD_MODEL = Path(__file__).parent.parent / "scripts" / "best.torch"

IMAGE_SIZE = 64
N_SAMPLES = 4


def write_color_mask(path: Path, color_map: ColorMap, rng):
    labels = rng.integers(0, len(color_map), size=(IMAGE_SIZE, IMAGE_SIZE))
    out = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    for spec in color_map:
        out[labels == spec.label] = spec.color
    Image.fromarray(out).save(path)


@pytest.fixture(scope="module")
def data_dirs(tmp_path_factory):
    """N_SAMPLES images + main masks; only the first half has a mask for the additional head."""
    root = tmp_path_factory.mktemp("multihead_data")
    dirs = {name: root / name for name in ["image", "mask", "add_mask"]}
    for d in dirs.values():
        d.mkdir()
    rng = np.random.default_rng(123)
    for i in range(N_SAMPLES):
        img = rng.integers(0, 255, size=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        Image.fromarray(img).save(dirs["image"] / f"page{i}.png")
        write_color_mask(dirs["mask"] / f"page{i}.png", MAIN_CMAP, rng)
        if i < N_SAMPLES // 2:
            write_color_mask(dirs["add_mask"] / f"page{i}.png", ADD_CMAP, rng)
    return dirs


def build_transforms(encoder_name: str, number_of_heads: int) -> PreprocessingTransforms:
    input_transforms = albumentations.Compose([
        GrayToRGBTransform(p=1.0),
        ColorMapTransform(p=1.0, color_map=MAIN_CMAP.to_albumentation_color_map())
    ])
    post_transforms = albumentations.Compose([
        NetworkEncoderTransform(encoder_name, p=1.0),
        ToTensorV2(p=1.0)
    ])
    transforms = PreprocessingTransforms(input_transform=input_transforms, post_transforms=post_transforms)
    if number_of_heads > 0:
        transforms.register_additional_targets([f"mask_head_{i}" for i in range(number_of_heads)])
    return transforms


def build_config(use_custom_model: bool, with_heads: bool) -> ModelConfiguration:
    add_heads = 1 if with_heads else 0
    add_classes = [len(ADD_CMAP)] if with_heads else []
    encoder_name = Preprocessingfunction.name if use_custom_model else "efficientnet-b3"
    transforms = build_transforms(encoder_name, add_heads)
    return ModelConfiguration(
        use_custom_model=use_custom_model,
        network_settings=None if use_custom_model else PredefinedNetworkSettings(
            classes=len(MAIN_CMAP), architecture=Architecture.UNET,
            add_number_of_heads=add_heads, add_classes=add_classes),
        custom_model_settings=CustomModelSettings(
            classes=len(MAIN_CMAP), encoder_filter=[16, 32, 64, 128], decoder_filter=[16, 32, 64, 128],
            attention_encoder_filter=[16, 32, 64, 128], attention=True,
            add_number_of_heads=add_heads, add_classes=add_classes) if use_custom_model else None,
        color_map=MAIN_CMAP,
        preprocessing_settings=ProcessingSettings(transforms=transforms.to_dict(), input_padding_value=32, rgb=True),
        additional_color_maps=[ADD_CMAP] if with_heads else None,
    )


def make_dataset(data_dirs, config: ModelConfiguration) -> MaskDataset:
    number_of_heads, _ = config.head_config()
    df = dirs_to_pandaframe(
        [str(data_dirs["image"])], [str(data_dirs["mask"])],
        additional_masks_dirs=[[str(data_dirs["add_mask"])]] if number_of_heads else None)
    transforms = PreprocessingTransforms.from_dict(config.preprocessing_settings.transforms)
    if number_of_heads > 0:
        transforms.register_additional_targets([f"mask_head_{i}" for i in range(number_of_heads)])
    return MaskDataset(df, transforms=transforms, additional_color_maps=config.additional_color_maps)


def test_dataset_marks_missing_additional_masks(data_dirs):
    config = build_config(use_custom_model=True, with_heads=True)
    dataset = make_dataset(data_dirs, config)
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    batches = list(loader)
    assert all(len(b) == 4 for b in batches)
    labeled = [b for b in batches if b[2][0].numel() > 0]
    unlabeled = [b for b in batches if b[2][0].numel() == 0]
    assert len(labeled) == N_SAMPLES // 2
    assert len(unlabeled) == N_SAMPLES - N_SAMPLES // 2
    # available additional targets are label-encoded with their own color map
    assert int(labeled[0][2][0].max()) < len(ADD_CMAP)


@pytest.mark.parametrize("use_custom_model", [True, False], ids=["custom", "predefined"])
def test_train_save_reload_predict_multihead(data_dirs, tmp_path, use_custom_model):
    config = build_config(use_custom_model=use_custom_model, with_heads=True)
    network = ModelBuilderMeta(config, "cpu").get_model()
    dataset = make_dataset(data_dirs, config)
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    model = network.model
    head_params_before = [p.detach().clone() for p in
                          (model.heads[0] if not use_custom_model else model.add_heads[0]).parameters()]

    number_of_heads, add_classes = config.head_config()
    mw = ModelWriterCallback(network, config, save_path=tmp_path)
    trainer = NetworkTrainer(network, NetworkTrainSettings(classes=len(MAIN_CMAP),
                                                           additional_heads=number_of_heads,
                                                           additional_classes=add_classes), "cpu",
                             callbacks=[mw])
    trainer.train_epochs(train_loader=loader, val_loader=loader, n_epoch=1, lr_schedule=None)

    head_params_after = list((model.heads[0] if not use_custom_model else model.add_heads[0]).parameters())
    assert any(not torch.equal(before, after.detach())
               for before, after in zip(head_params_before, head_params_after)), \
        "additional head weights did not train"

    # reload the saved model and predict
    loaded = ModelBuilderLoad.from_disk(mw.get_best_model_path(), device="cpu")
    loaded_network = loaded.get_model()
    predictor = NetworkPredictor.from_model_config(loaded_network, loaded.get_model_configuration())
    result = predictor.predict_image(SourceImage.load(data_dirs["image"] / "page0.png"))
    assert result.probability_map.shape[-1] == len(MAIN_CMAP)
    assert isinstance(result.other_probability_map, list) and len(result.other_probability_map) == 1
    assert result.other_probability_map[0].shape[-1] == len(ADD_CMAP)


@pytest.mark.skipif(not OLD_MODEL.exists(), reason="old single-head checkpoint not available")
def test_old_single_head_checkpoint_loads_unchanged(data_dirs):
    loaded = ModelBuilderLoad.from_disk(OLD_MODEL, device="cpu")
    network = loaded.get_model()
    # strict load must be bit-exact
    checkpoint = torch.load(OLD_MODEL, map_location="cpu")
    for key, value in network.model.state_dict().items():
        assert torch.equal(value, checkpoint[key])
    config = loaded.get_model_configuration()
    assert config.head_config() == (0, [])
    predictor = NetworkPredictor.from_model_config(network, config)
    result = predictor.predict_image(SourceImage.load(data_dirs["image"] / "page0.png"))
    assert result.other_probability_map is None
    assert result.probability_map.shape[-1] == len(config.color_map)


@pytest.mark.skipif(not OLD_MODEL.exists(), reason="old single-head checkpoint not available")
def test_warm_start_multihead_from_old_checkpoint(data_dirs):
    old_config = ModelFile.from_file(OLD_MODEL.with_suffix(".json")).model_configuration
    # same architecture as the old model, but with one additional head
    old_config.network_settings.add_number_of_heads = 1
    old_config.network_settings.add_classes = [len(ADD_CMAP)]
    old_config.additional_color_maps = [ADD_CMAP]

    network = ModelBuilderMeta(old_config, "cpu").get_model()
    load_weights_into(network, OLD_MODEL, "cpu")

    checkpoint = torch.load(OLD_MODEL, map_location="cpu")
    state = network.model.state_dict()
    # shared weights (now under the 'model.' prefix of MultiHeadNetwork) taken from the checkpoint
    for key, value in checkpoint.items():
        assert torch.equal(state[f"model.{key}"], value)
    # the wrapped model exposes the extra head and returns a tuple
    output = network.model(torch.zeros((1, 3, 64, 64)))
    assert isinstance(output, tuple)
    assert output[0].shape[1] == len(old_config.color_map)
    assert output[1][0].shape[1] == len(ADD_CMAP)
