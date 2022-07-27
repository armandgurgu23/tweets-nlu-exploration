from omegaconf import DictConfig, OmegaConf
import logging
from trainers import trainers_registry


log = logging.getLogger(__name__)


def main_experiment(cfg: DictConfig):
    log.info(f'Running an experiment! Experiment configuration used: ')
    log.info(OmegaConf.to_yaml(cfg))
    experiment_trainer = trainers_registry[cfg.experiment.trainer_type]
    experiment_trainer(experiment_config=cfg.experiment)()
    return
