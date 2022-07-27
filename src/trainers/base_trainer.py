from abc import ABC, abstractmethod
from omegaconf import DictConfig


class BaseTrainer(ABC):

    def __init__(self, experiment_config: DictConfig) -> None:
        self.config = experiment_config
        self.train_loader = self.initialize_dataset_loader(
            self.config.data.train.filepath)
        self.valid_loader = self.initialize_dataset_loader(
            self.config.data.valid.filepath)
        if self.config.evaluate_on_test_data:
            self.test_loader = self.initialize_dataset_loader(
                self.config.data.test.filepath)

    def __call__(self):
        self.train_model()
        if self.config.evaluate_on_test_data:
            self.evaluate_model()
        return

    @abstractmethod
    def train_model(self):
        pass

    @abstractmethod
    def evaluate_model(self):
        pass

    @abstractmethod
    def initialize_dataset_loader(self, dataset_path: str):
        pass


class SklearnTrainer(BaseTrainer):

    def initialize_dataset_loader(self, dataset_path: str):
        raise NotImplementedError(
            "start implementing setup for sklearn dataset loader!!")

    def train_model(self):
        raise NotImplementedError("Start implementing sklearn training!")

    def evaluate_model(self):
        raise NotImplementedError('Start implementing sklearn evaluation!')
