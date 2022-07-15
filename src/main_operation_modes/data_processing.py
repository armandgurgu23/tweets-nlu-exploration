from omegaconf import DictConfig
import logging
from data.data_readers import get_dataset
from data.data_processors import process_dataset
from data.data_splitters import split_dataset_into_ml_datasets, write_ml_datasets_to_disk
import os

log = logging.getLogger(__name__)


def main_data_processing(cfg: DictConfig):
    log.info(f'Running data processing mode on dataset {cfg.data.data_path}!')
    dataset = get_dataset(cfg.data.data_path, cfg.data.pandas_read_csv_config)
    processed_dataset = process_dataset(
        dataset, cfg.data.data_processing_config)
    # Add code to split datasets into train-valid-test splits
    # and to write them to disk here.
    ml_datasets = split_dataset_into_ml_datasets(
        processed_dataset, cfg.seed, cfg.data.data_split_config)
    write_ml_datasets_to_disk(
        ml_datasets, cfg.data.data_path, cfg.data.pandas_to_csv_config)
    log.info(
        f"Finished processing and splitting dataset {cfg.data.data_path}! Splits dumped at: {os.getcwd()}!")
    return
