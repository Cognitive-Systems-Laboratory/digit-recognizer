import hydra
import omegaconf
import wandb

from digitrec.trainer import setup_trainer


@hydra.main(version_base="1.3", config_path="configs", config_name="train")
def main(cfg: omegaconf.DictConfig) -> None:
    wandb.init(
        project=cfg.logging.project,
        name=cfg.logging.name,
        config=omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
    )
    trainer = setup_trainer(config=cfg)
    trainer.fit()


if __name__=="__main__":
    main()
