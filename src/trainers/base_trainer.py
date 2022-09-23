from abc import ABC, abstractmethod
from typing import Any, List
from omegaconf import DictConfig
from models import models_registry
from pandas import read_csv
from numpy import mean as np_mean
import logging

log = logging.getLogger(__name__)


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

    def get_all_labels_in_dataset(self, train_loader: Any) -> List[str]:
        all_labels_in_dataset = []
        for pd_batch in train_loader:
            all_labels_in_dataset += pd_batch['label'].tolist()
        return sorted(set(all_labels_in_dataset))

    def train_model(self):
        minibatch_count = 0
        unique_labels = self.get_all_labels_in_dataset(self.train_loader)
        training_losses = []
        for current_epoch in range(self.config.trainer.num_epochs):
            self.train_loader = self.initialize_dataset_loader(
                self.config.data.train.filepath, self.config.data.train.reader_config)
            self.valid_loader = self.initialize_dataset_loader(
                self.config.data.valid.filepath, self.config.data.valid.reader_config)
            for pd_batch in self.train_loader:
                text_batch, labels_batch = pd_batch['content_cleaned'].tolist(
                ), pd_batch['label'].tolist()
                curr_train_loss = self.model_to_train.run_training_step(
                    text_batch, labels_batch, unique_labels=unique_labels)
                training_losses.append(curr_train_loss)
                minibatch_count += 1
                if minibatch_count % self.config.trainer.log_train_loss_per_updates == 0:
                    log.info(
                        f'Current train loss (epoch {current_epoch+1}/{self.config.trainer.num_epochs}): {np_mean(training_losses)}')
                    training_losses = []
                if minibatch_count % self.config.trainer.save_model_per_updates == 0:
                    self.model_to_train.save_model(
                        output_path=f"ckpt_{minibatch_count}")
        log.info('Finished training model! Storing final model!')
        self.model_to_train.save_model(
            output_path=f"final_model_ckpt")
        return

    def evaluate_model(self):
        raise NotImplementedError(
            'Start implementing sklearn evaluation method on test set!!')
