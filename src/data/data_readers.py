import os
from omegaconf import DictConfig
import pandas as pd


def read_csv_dataset(dataset_path: str, pandas_read_csv_config: DictConfig) -> pd.DataFrame:
    return pd.read_csv(filepath_or_buffer=dataset_path, **pandas_read_csv_config)


dataset_readers = {
    'csv': read_csv_dataset,

}


def get_dataset(dataset_path: str, reader_config: DictConfig):
    if os.path.isfile(dataset_path):
        dataset_ext = dataset_path.split('/')[-1].split('.')[-1]
        reader_method = dataset_readers[dataset_ext]
        return reader_method(dataset_path, reader_config)
    else:
        raise NotImplementedError(f'Dataset can only be read in as a file!')
