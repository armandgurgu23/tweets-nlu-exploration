from abc import ABC, abstractmethod


class MLModel(ABC):

    @abstractmethod
    def initialize_model(self):
        pass

    @abstractmethod
    def save_model(self):
        pass

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def run_training_step(self):
        pass

    @abstractmethod
    def run_validation_step(self):
        pass
