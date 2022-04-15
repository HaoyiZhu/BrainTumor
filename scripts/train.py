import hydra
import pytorch_lightning as pl

from brain_tumor.learn import BrainTumorTrainer


@hydra.main(config_path="configs", config_name="baseline")
def main(cfg):
    pl.seed_everything(cfg.seed)
    trainer = BrainTumorTrainer(cfg)
    trainer.fit()


if __name__ == "__main__":
    main()