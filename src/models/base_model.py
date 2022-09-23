from abc import ABC, abstractmethod
from sklearn.metrics import log_loss
from omegaconf import DictConfig
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from typing import Any, Dict, List
from os import makedirs, path
from joblib import dump as jl_dump
from joblib import load as jl_load
from numpy import argmax, mean
from sklearn.metrics import precision_score, recall_score, accuracy_score


class MLModel(ABC):

    def __init__(self, model_config: DictConfig) -> None:
        self.config = model_config
        self.model = self.initialize_model(self.config)
        self.vectorizer = self.initialize_data_vectorizer(
            self.config.vectorizer)

    @abstractmethod
    def initialize_model(self, model_config: DictConfig) -> Any:
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
    def run_training_step(self, text_batch: Any, label_batch: Any, **kwargs: Dict) -> float:
        pass

    @abstractmethod
    def run_validation_epoch(self, validation_reader: Any):
        pass

    @abstractmethod
    def forward_pass(self):
        pass


sklearn_vectorizers = {
    'hashing': HashingVectorizer
}


class SklearnModel(MLModel):

    def initialize_model(self, model_config: DictConfig) -> SGDClassifier:
        if model_config.use_sgd_architecture:
            return SGDClassifier(**model_config.architecture_config)
        else:
            raise NotImplementedError(
                f'So far only minibatch training supported for sklearn model!')

    def initialize_data_vectorizer(self, vectorizer_config: DictConfig) -> HashingVectorizer:
        vectorizer = sklearn_vectorizers[vectorizer_config.name]
        return vectorizer(**vectorizer_config.config)

    def save_model(self, output_path: str) -> None:
        makedirs(output_path, exist_ok=True)
        jl_dump(self.model, path.join(output_path, 'model.joblib'))
        return

    def load_model(self, load_path: str) -> "MLModel":
        return jl_load(load_path)

    def encode_labels_to_vectors(self, label_batch: Any, all_labels: List[str]):
        return [all_labels.index(current_label) for current_label in label_batch]

    def run_training_step(self, text_batch: Any, label_batch: Any, **kwargs: Dict):
        text_vectors = self.vectorizer.fit_transform(text_batch)
        label_vectors = self.encode_labels_to_vectors(
            label_batch, kwargs['unique_labels'])
        self.model.partial_fit(text_vectors, label_vectors, classes=list(
            range(len(kwargs['unique_labels']))))
        prediction_vectors = self.model.predict_proba(text_vectors)
        train_loss = log_loss(y_true=label_vectors, y_pred=prediction_vectors, labels=list(
            range(len(kwargs['unique_labels']))))
        return train_loss

    def run_validation_epoch(self, validation_reader: Any, metrics_config: Dict, **kwargs: Dict):
        validation_losses = []
        all_preds = []
        all_gt = []
        for pd_batch in validation_reader:
            text_batch, labels_batch = pd_batch['content_cleaned'].tolist(
            ), pd_batch['label'].tolist()
            text_vectors = self.vectorizer.fit_transform(text_batch)
            label_vectors = self.encode_labels_to_vectors(
                labels_batch, kwargs['unique_labels'])
            prediction_vectors = self.model.predict_proba(
                text_vectors)
            prediction_indexes = argmax(prediction_vectors, axis=-1).tolist()
            curr_valid_loss = log_loss(y_true=label_vectors, y_pred=prediction_vectors, labels=list(
                range(len(kwargs['unique_labels']))))
            validation_losses.append(curr_valid_loss)
            all_preds += prediction_indexes
            all_gt += label_vectors
        validation_metrics = {
            'valid_precision': precision_score(y_true=all_gt, y_pred=all_preds, **metrics_config),
            'valid_recall': recall_score(y_true=all_gt, y_pred=all_preds, **metrics_config),
            'valid_accuracy': accuracy_score(y_true=all_gt, y_pred=all_preds),
            'valid_loss': mean(validation_losses)
        }
        return validation_metrics

    def forward_pass(self):
        raise NotImplementedError('Fill logic for sklearn.forward_pass!')
