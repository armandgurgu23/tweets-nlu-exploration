from abc import ABC, abstractmethod
from omegaconf import DictConfig
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from typing import Any


class MLModel(ABC):

    def __init__(self, model_config: DictConfig) -> None:
        self.config = model_config
        self.model = self.initialize_model(self.config)
        self.vectorizer = self.initialize_data_vectorizer(
            self.config.vectorizer)

    @abstractmethod
    def initialize_model(self, model_config: DictConfig) -> "MLModel":
        pass

    @abstractmethod
    def initialize_data_vectorizer(self, vectorizer_config: DictConfig) -> Any:
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


sklearn_vectorizers = {
    'hashing': HashingVectorizer
}


class SklearnModel(MLModel):

    def initialize_model(self, model_config: DictConfig) -> "MLModel":
        if model_config.use_sgd_architecture:
            return SGDClassifier(**model_config.architecture_config)
        else:
            raise NotImplementedError(
                f'So far only minibatch training supported for sklearn model!')

    def initialize_data_vectorizer(self, vectorizer_config: DictConfig) -> HashingVectorizer:
        vectorizer = sklearn_vectorizers[vectorizer_config.name]
        return vectorizer(**vectorizer_config.config)

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
