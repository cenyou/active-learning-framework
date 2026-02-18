from typing import Optional
from alef.configs.models.base_model_config import BaseModelConfig
# config for PFN model of our API

class BasicTabPFNModelConfig(BaseModelConfig):
    device: str = "cpu"
    name: str = "TabPFNModel"

class TabPFNModelGPUConfig(BasicTabPFNModelConfig):
    device: str = "cuda"
    name: str = "TabPFNModel_CUDA"

