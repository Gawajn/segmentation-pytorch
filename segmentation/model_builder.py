import abc
import re
from pathlib import Path
from typing import Dict, Union

import loguru
import torch

from segmentation.network import Network
from segmentation.settings import CustomModelSettings, PredefinedNetworkSettings, ModelConfiguration, ModelFile, ProcessingSettings

# state_dict keys belonging to additional heads: MultiHeadNetwork registers them as
# head_{i}.*, the custom models as add_heads.{i}.*
ADDITIONAL_HEAD_KEY = re.compile(r"^(head_\d+\.|add_heads\.)")


def is_additional_head_key(key: str) -> bool:
    return bool(ADDITIONAL_HEAD_KEY.match(key))


def adapt_state_dict_keys(state_dict: Dict[str, torch.Tensor], model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Remap checkpoint keys between plain models and MultiHeadNetwork-wrapped models.

    Wrapping a plain smp model in MultiHeadNetwork prefixes its keys with 'model.'.
    """
    from segmentation.multi_head_network import MultiHeadNetwork
    target_is_wrapped = isinstance(model, MultiHeadNetwork)
    state_is_wrapped = any(k.startswith("model.") for k in state_dict)
    if target_is_wrapped and not state_is_wrapped:
        return {k if is_additional_head_key(k) else f"model.{k}": v for k, v in state_dict.items()}
    if not target_is_wrapped and state_is_wrapped:
        return {k[len("model."):]: v for k, v in state_dict.items() if k.startswith("model.")}
    return dict(state_dict)


def load_weights_into(network: Network, model_weights: Union[Path, str], device) -> Network:
    """Warm-start: load all compatible weights from a checkpoint into an already built network.

    Keys are remapped if the head wrapping differs, keys missing from the checkpoint or with
    mismatching shapes are left at their (random) initialization. Used to fine-tune an old
    single-head model into a new architecture with additional heads.
    """
    state_dict = torch.load(Path(model_weights), map_location=torch.device(device))
    adapted = adapt_state_dict_keys(state_dict, network.model)
    model_state = network.model.state_dict()
    filtered = {}
    skipped = []
    for key, value in adapted.items():
        if key in model_state and model_state[key].shape == value.shape:
            filtered[key] = value
        else:
            skipped.append(key)
    missing, _ = network.model.load_state_dict(filtered, strict=False)
    if skipped:
        loguru.logger.warning(f"Warm-start from {model_weights}: skipped incompatible checkpoint keys: {skipped}")
    if missing:
        loguru.logger.info(f"Warm-start from {model_weights}: randomly initialized parameters: {list(missing)}")
    return network


class ModelBuilderBase(abc.ABC):
    @abc.abstractmethod
    def get_model(self) -> Network:
        pass


class ModelBuilderCustom(ModelBuilderBase):
    def __init__(self, custom_model_settings: CustomModelSettings, preprocessing_settings: ProcessingSettings, device):
        self.custom_model_settings = custom_model_settings
        self.preprocessing_settings = preprocessing_settings
        self.device = device

    def get_model(self) -> Network:
        from segmentation.custom_model import CustomModel
        kwargs = self.custom_model_settings.get_kwargs()
        model = CustomModel(self.custom_model_settings.type)()(**kwargs)
        model.to(self.device)
        return Network(model, self.preprocessing_settings, self.device)


class ModelBuilderPredefined(ModelBuilderBase):
    def __init__(self, settings: PredefinedNetworkSettings, preprocessing_settings: ProcessingSettings, device):
        self.settings = settings
        self.preprocessing_settings = preprocessing_settings
        self.device = device

    def get_model(self) -> Network:
        model_params = self.settings.architecture.get_architecture_params()
        model_params['classes'] = self.settings.classes

        model_params['encoder_name'] = self.settings.encoder

        if not self.settings.use_batch_norm_layer:
            model_params['decoder_use_batchnorm'] = False

        if 'decoder_channels' in model_params:
            model_params['decoder_channels'] = self.settings.decoder_channel

        if 'encoder_depth' in model_params:
            model_params['encoder_depth'] = self.settings.encoder_depth

        if self.settings.architecture in [self.settings.architecture.DeepLabV3Plus, self.settings.architecture.DeepLabV3, self.settings.architecture.PAN]:
            model_params['decoder_channels'] = self.settings.decoder_channel[0]

        kwargs = {k: v for k, v in model_params.items() if v is not None}
        model = self.settings.architecture.get_architecture()(**kwargs)

        # Only Unet supported yet
        if self.settings.add_number_of_heads > 0:
            from segmentation.multi_head_network import MultiHeadNetwork
            output_channels = kwargs.get('decoder_channels', None)[-1] if kwargs.get('decoder_channels', None)  is not None else None
            model = MultiHeadNetwork(model, self.settings.add_number_of_heads, kwargs.get('activation', None), 1 if kwargs.get('upsampling', None) is None else kwargs.get('upsampling', None), add_classes=self.settings.add_classes, out_channels=output_channels)

        model.to(self.device)
        return Network(model, self.preprocessing_settings, self.device)



class ModelBuilderMeta(ModelBuilderBase):
    def __init__(self, model_config: ModelConfiguration, device):
        self.model_config = model_config
        self.device = device

    def get_model(self) -> Network:
        if self.model_config.use_custom_model:
            network = ModelBuilderCustom(self.model_config.custom_model_settings,
                                         self.model_config.preprocessing_settings,
                                         self.device).get_model()
        else:
            network = ModelBuilderPredefined(self.model_config.network_settings,
                                             self.model_config.preprocessing_settings,
                                             self.device).get_model()
        return network


class ModelBuilderLoad(ModelBuilderBase):
    def __init__(self, model_file: ModelFile, model_weights: Path, device):
        self.model_file = model_file
        self.device = device
        self.weights_path = model_weights

    def get_model(self) -> Network:
        network = ModelBuilderMeta(self.model_file.model_configuration, self.device).get_model()
        state_dict = torch.load(self.weights_path, map_location=torch.device(self.device))
        try:
            network.model.load_state_dict(state_dict)
        except RuntimeError:
            adapted = adapt_state_dict_keys(state_dict, network.model)
            missing, unexpected = network.model.load_state_dict(adapted, strict=False)
            missing_non_head = [k for k in missing if not is_additional_head_key(k)]
            if unexpected or missing_non_head:
                raise RuntimeError(
                    f"Checkpoint {self.weights_path} does not match the model built from its configuration. "
                    f"Missing keys: {missing_non_head}, unexpected keys: {list(unexpected)}")
            if missing:
                loguru.logger.warning(
                    f"Checkpoint {self.weights_path} contains no weights for the additional heads, "
                    f"they are randomly initialized: {list(missing)}")
        return network

    def get_model_configuration(self) -> ModelConfiguration:
        return self.model_file.model_configuration

    @classmethod
    def from_disk(cls, model_weights: Union[Path, str], device):
        if type(model_weights) is str:
            model_weights = Path(model_weights)
        json_f = model_weights.with_suffix(".json")
        mf = ModelFile.from_file(json_f)
        return cls(mf, model_weights, device)
