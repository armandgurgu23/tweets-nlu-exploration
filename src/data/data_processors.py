from omegaconf import DictConfig
import pandas as pd
from fasttext import load_model


def remove_non_english_examples(dataset: pd.DataFrame, filtering_config: DictConfig):
    print(dataset)
    print(filtering_config)
    raise NotImplementedError('Will get here soon!?')


def remove_hashtag_labels_from_tweets(dataset: pd.DataFrame):
    print(dataset)
    raise NotImplementedError('FDSKFDJSLFJLSK!?!?!?')


def remove_hashtag_label_from_single_tweet(row: pd.DataFrame):
    pass


def process_dataset(dataset: pd.DataFrame, filtering_config: DictConfig) -> pd.DataFrame:
    dataset = remove_hashtag_labels_from_tweets(dataset)
    dataset = remove_non_english_examples(dataset, filtering_config)
    return dataset
