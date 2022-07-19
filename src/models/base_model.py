from abc import ABC, abstractmethod
from omegaconf import DictConfig


class MLModel(ABC):

    def __init__(self, model_config: DictConfig) -> None:
        self.config = model_config
        self.model = self.initialize_model(self.config)

    @abstractmethod
    def initialize_model(self, model_config: DictConfig) -> "MLModel":
        pass

    @abstractmethod
    def save_model(self, output_path: str) -> None:
        pass

    @abstractmethod
    def load_model(self, load_path: str) -> "MLModel":
        pass

    @abstractmethod
    def run_training_step(self):
        pass

    @abstractmethod
    def run_validation_step(self):
        pass

    @abstractmethod
    def forward_pass(self):
        pass
