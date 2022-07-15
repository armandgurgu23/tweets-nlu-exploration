from omegaconf import DictConfig
import logging
from data.data_readers import get_dataset

log = logging.getLogger(__name__)


def main_data_processing(cfg: DictConfig):
    log.info(f'Running data processing mode on dataset {cfg.data.data_path}!')
    dataset = get_dataset(cfg.data.data_path, cfg.data.pandas_read_csv_config)
    print(dataset)
    print(dataset.columns)
    print(dataset.lang.value_counts())
    print(dataset['query'].value_counts())
    return
