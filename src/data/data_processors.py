from omegaconf import DictConfig
import pandas as pd
from fasttext import load_model
from fasttext import FastText
from typing import Tuple
import logging

log = logging.getLogger(__name__)


def remove_non_english_examples(dataset: pd.DataFrame, filtering_config: DictConfig):
    fasttext_model = load_model(filtering_config.model_path)
    fasttext_preds = dataset['content_cleaned'].apply(
        lambda content: get_fasttext_prediction(content, fasttext_model, filtering_config.confidence))
    dataset['fasttext_lang'] = [pred_tuple[0] for pred_tuple in fasttext_preds]
    dataset['fasttext_conf'] = [pred_tuple[1] for pred_tuple in fasttext_preds]
    filtered_dataset = dataset[(dataset.fasttext_lang == 'en') & (
        dataset.fasttext_conf >= filtering_config.confidence)]
    proportion_dataset_remaining = round(
        (filtered_dataset.shape[0] / dataset.shape[0]) * 100.0, 2)
    log.info(
        f"Filtering with threshold {filtering_config.confidence} for English complete! Resulting dataset has {filtered_dataset.shape[0]}/{dataset.shape[0]} rows ({proportion_dataset_remaining}%)!")
    return filtered_dataset


def get_fasttext_prediction(input_text: str, fasttext_model: FastText._FastText, threshold: float) -> Tuple[str, float]:
    raw_prediction_tuple = fasttext_model.predict(
        input_text.replace('\n', ''), threshold=threshold)
    if raw_prediction_tuple[0]:
        predicted_language = raw_prediction_tuple[0][0].split('__')[-1]
        predicted_conf = raw_prediction_tuple[1][0]
        return predicted_language, predicted_conf
    else:
        return 'not-confident', 0.0


def remove_hashtag_labels_from_tweets(dataset: pd.DataFrame):
    dataset['content_cleaned'] = dataset.apply(
        lambda row: remove_hashtag_label_from_single_tweet(row), axis=1)
    return dataset


def remove_hashtag_label_from_single_tweet(row: pd.DataFrame):
    hashtag_in_query = row['query'].split()[0]
    cleaned_content = row['content'].replace(hashtag_in_query, '')
    cleaned_content = cleaned_content.replace(
        hashtag_in_query[0] + hashtag_in_query[1:].capitalize(), '')
    cleaned_content = cleaned_content.replace(hashtag_in_query[1:], '')
    cleaned_content = cleaned_content.replace(
        hashtag_in_query[1:].capitalize(), '')
    return cleaned_content.strip()


def process_dataset(dataset: pd.DataFrame, filtering_config: DictConfig) -> pd.DataFrame:
    dataset = remove_hashtag_labels_from_tweets(dataset)
    if filtering_config.apply_english_filtering:
        log.info(f'Filtering non-English tweets using FastText model!')
        dataset = remove_non_english_examples(dataset, filtering_config)
        dataset = dataset.drop(
            columns=['fasttext_lang', 'fasttext_conf', 'lang'])
        log.info(f"Done filtering non-English tweets!")
    return dataset
