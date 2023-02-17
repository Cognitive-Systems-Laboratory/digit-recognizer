import numpy as np
import hydra
import omegaconf
import wandb

@hydra.main(version_base="1.3", config_path="configs", config_name="train")
def main(cfg: omegaconf.DictConfig) -> None:
    wandb.init(
        project="test-project",
        name="test-run",
        config=omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
    )
    for idx, val in enumerate(np.random.random(size=20)):
        wandb.log({"foo": val}, step=idx)

if __name__=="__main__":
    main()