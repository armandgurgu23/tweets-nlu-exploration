import hydra
from omegaconf import DictConfig, OmegaConf
from main_operation_modes.data_processing import main_data_processing
from main_operation_modes.experiment import main_experiment


@hydra.main(version_base='1.1', config_path="config", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    # Explicitly resolve interpolations in config in order to pass
    # paths from environment variables.
    OmegaConf.resolve(cfg)
    if cfg.main_operation_mode == 'data_processing':
        main_data_processing(cfg)
    elif cfg.main_operation_mode == 'experiment':
        main_experiment(cfg)
    else:
        raise NotImplementedError(
            f"Operation mode {cfg.main_operation_mode} not supported!")


if __name__ == "__main__":
    main()
