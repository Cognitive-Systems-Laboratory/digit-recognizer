import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="configs", config_name="config")
def test(cfg: DictConfig) -> None:
    loss_fn = hydra.utils.instantiate(cfg.loss)
    breakpoint()
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    test()