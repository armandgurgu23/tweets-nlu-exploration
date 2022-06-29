import hydra
from omegaconf import DictConfig


@hydra.main(version_base='1.1', config_path="config", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    print('\n\n Hello Hydra\n')
    print(cfg)
    return


if __name__ == "__main__":
    main()
