from typing import Dict
from sklearn.model_selection import train_test_split
import pandas as pd
from omegaconf import DictConfig


def split_dataset_into_ml_datasets(dataset: pd.DataFrame, seed: int, data_split_config: DictConfig):
    if data_split_config.do_stratified_split:
        train_valid_datasets, test_dataset = train_test_split(
            dataset, test_size=data_split_config.test_split_proportion, random_state=seed, stratify=dataset['label'])
        train_dataset, valid_dataset = train_test_split(
            train_valid_datasets, test_size=data_split_config.valid_split_proportion, random_state=seed, stratify=train_valid_datasets['label'])
    else:
        train_valid_datasets, test_dataset = train_test_split(
            dataset, test_size=data_split_config.test_split_proportion, random_state=seed)
        train_dataset, valid_dataset = train_test_split(
            train_valid_datasets, test_size=data_split_config.valid_split_proportion, random_state=seed)
    return {
        'train': train_dataset,
        'valid': valid_dataset,
        'test': test_dataset
    }


def write_ml_datasets_to_disk(ml_datasets: Dict[str, pd.DataFrame], dataset_path: str, write_config: DictConfig) -> None:
    dataset_filename, dataset_ext = dataset_path.split(
        '/')[-1].split('.')[0], dataset_path.split('/')[-1].split('.')[1]
    for dataset_type in ml_datasets:
        curr_dataset = ml_datasets[dataset_type]
        output_name = f"{dataset_filename}_processed_{dataset_type}.{dataset_ext}"
        curr_dataset.to_csv(path_or_buf=output_name, **write_config)
    return
