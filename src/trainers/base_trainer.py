from abc import ABC, abstractmethod
from omegaconf import DictConfig
from models import models_registry
from pandas import read_csv


class BaseTrainer(ABC):

    def __init__(self, experiment_config: DictConfig) -> None:
        self.config = experiment_config
        self.train_loader = self.initialize_dataset_loader(
            self.config.data.train.filepath, self.config.data.train.reader_config)
        self.valid_loader = self.initialize_dataset_loader(
            self.config.data.valid.filepath, self.config.data.valid.reader_config)
        if self.config.evaluate_on_test_data:
            self.test_loader = self.initialize_dataset_loader(
                self.config.data.test.filepath, self.config.data.test.reader_config)
        self.model_to_train = models_registry[self.config.model_type](
            model_config=self.config.model)

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
    def initialize_dataset_loader(self, dataset_path: str, reader_config: DictConfig):
        pass


class SklearnTrainer(BaseTrainer):

    def initialize_dataset_loader(self, dataset_path: str, reader_config: DictConfig):
        return read_csv(dataset_path, **reader_config)

    def train_model(self):
        raise NotImplementedError(
            'Implement logic for training sklearn model!')

    def evaluate_model(self):
        raise NotImplementedError(
            'Start implementing sklearn evaluation method!')
