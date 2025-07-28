import abc
from pathlib import Path
from typing import Union

import torch

from segmentation.network import Network
from segmentation.settings import CustomModelSettings, PredefinedNetworkSettings, ModelConfiguration, ModelFile, ProcessingSettings


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

        if 'decoder_use_batchnorm' in model_params and self.settings.use_batch_norm_layer is False:
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
        if self.settings.number_of_heads > 1:
            from segmentation.multi_head_neatwork import MultiHeadNetwork
            output_channels = kwargs.get('decoder_channels', None)[-1] if kwargs.get('decoder_channels', None)  is not None else None
            model = MultiHeadNetwork(model, self.settings.number_of_heads, kwargs.get('activation', None), 1 if kwargs.get('upsampling', None) is None else kwargs.get('upsampling', None), add_classes=self.settings.add_classes, out_channels=output_channels)

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
        network.model.load_state_dict(torch.load(self.weights_path, map_location=torch.device(self.device)))
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
