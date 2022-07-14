from calendar import c
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base='1.1', config_path="config", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    # Explicitly resolve interpolations in config in order to pass
    # paths from environment variables.
    OmegaConf.resolve(cfg)
    print(cfg)
    return


if __name__ == "__main__":
    main()
