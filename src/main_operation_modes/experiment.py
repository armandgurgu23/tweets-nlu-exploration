from omegaconf import DictConfig, OmegaConf
import logging


log = logging.getLogger(__name__)


def main_experiment(cfg: DictConfig):
    log.info(f'Running an experiment! Experiment configuration used: ')
    print(OmegaConf.to_yaml(cfg))
    return
