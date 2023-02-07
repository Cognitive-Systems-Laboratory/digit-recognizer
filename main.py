import hydra
import omegaconf
from digitrec.trainer import setup_trainer


@hydra.main(version_base="1.3", config_path="configs", config_name="train")
def main(cfg: omegaconf.DictConfig) -> None:
    trainer = setup_trainer(config=cfg)
    trainer.fit()


if __name__=="__main__":
    main()
