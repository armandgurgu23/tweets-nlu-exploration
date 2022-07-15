from calendar import c
import hydra
from omegaconf import DictConfig, OmegaConf
from main_operation_modes.data_processing import main_data_processing


@hydra.main(version_base='1.1', config_path="config", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    # Explicitly resolve interpolations in config in order to pass
    # paths from environment variables.
    OmegaConf.resolve(cfg)
    if cfg.main_operation_mode == 'data_processing':
        main_data_processing(cfg)
    else:
        raise NotImplementedError(
            f"Operation mode {cfg.main_operation_mode} not supported!")


if __name__ == "__main__":
    main()
