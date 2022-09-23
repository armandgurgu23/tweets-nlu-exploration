from omegaconf import DictConfig
from hydra.utils import get_original_cwd


class MLflowTracker(object):
    def __init__(self, tracker_config: DictConfig) -> None:
        self.config = tracker_config
        print(self.config)
        raise NotImplementedError('Implement MLflow tracker logic!?')
