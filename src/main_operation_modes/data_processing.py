from omegaconf import DictConfig
import logging
from data.data_readers import get_dataset
from data.data_processors import process_dataset

log = logging.getLogger(__name__)


def main_data_processing(cfg: DictConfig):
    log.info(f'Running data processing mode on dataset {cfg.data.data_path}!')
    dataset = get_dataset(cfg.data.data_path, cfg.data.pandas_read_csv_config)
    processed_dataset = process_dataset(
        dataset, cfg.data.data_processing_config)
    # Add code to split datasets into train-valid-test splits
    # and to write them to disk here.
    return
