from abc import ABC, abstractmethod


class BaseTrainer(ABC):

    @abstractmethod
    def train_model(self):
        pass

    @abstractmethod
    def evaluate_model(self):
        pass

    @abstractmethod
    def initialize_dataset_loader(self, dataset_path: str):
        pass
