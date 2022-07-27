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


class SklearnModel(MLModel):

    def initialize_model(self, model_config: DictConfig) -> "MLModel":
        print(model_config)
        raise NotImplementedError('Fill logic for sklearn.initialize_model!')

    def save_model(self, output_path: str) -> None:
        raise NotImplementedError('Fill logic for sklearn.save_model!')

    def load_model(self, load_path: str) -> "MLModel":
        raise NotImplementedError('Fill logic for sklearn.load_model!')

    def run_training_step(self):
        raise NotImplementedError('Fill logic for sklearn.run_training_step!')

    def run_validation_step(self):
        raise NotImplementedError(
            'Fill logic for sklearn.run_validation_step!')

    def forward_pass(self):
        raise NotImplementedError('Fill logic for sklearn.forward_pass!')
