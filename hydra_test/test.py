import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="configs", config_name="config")
def test(cfg: DictConfig) -> None:
    breakpoint()
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    test()