from __future__ import annotations

import numpy as np
from importlib import import_module
from omegaconf import DictConfig, OmegaConf

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

import brain_tumor.utils as U


class BrainTumorDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig, batch_size: int, num_workers: int = 0,) -> None:
        super().__init__()
        self.cfg = cfg

        self.batch_size = batch_size
        self.num_workers = num_workers

        if self.cfg.task == "classification":
            from brain_tumor.datasets import BraTClassificationDataset as BraTDataset
        elif self.cfg.task == "segmentation":
            from brain_tumor.datasets import BraTSegmentationDataset as BraTDataset
        else:
            raise NotImplementedError
        self.dataset = BraTDataset

    def setup(self, stage: str | None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = self.dataset(
                cfg=self.cfg,
                root=self.cfg.root,
                img_dim=self.cfg.img_dim,
                mri_type=self.cfg.mri_type,
                train=True,
            )
            self.val_dataset = self.dataset(
                cfg=self.cfg,
                root=self.cfg.root,
                img_dim=self.cfg.img_dim,
                mri_type=self.cfg.mri_type,
                train=False,
            )
        if stage == "test" or stage is None:
            self.test_dataset = self.dataset(
                cfg=self.cfg,
                root=self.cfg.root,
                img_dim=self.cfg.img_dim,
                mri_type=self.cfg.mri_type,
                train=False,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )


class BrainTumor(pl.LightningModule):
    def __init__(self, cfg: DictConfig, lr_cosine_steps_per_epoch: int = 1,) -> None:
        super().__init__()
        self.cfg = cfg

        self.lr = cfg.train.lr
        self.max_epochs = cfg.train.max_epochs
        self.weight_decay = cfg.train.weight_decay
        self.lr_cosine_min = cfg.train.scheduler.lr_cosine_min
        self.lr_cosine_epochs = cfg.train.scheduler.lr_cosine_epochs
        self.lr_cosine_warmup_epochs = cfg.train.scheduler.lr_cosine_warmup_epochs
        self.lr_cosine_steps_per_epoch = lr_cosine_steps_per_epoch

        model = U.build_model(cfg.model)
        self.model = U.load_checkpoint(model, checkpoint=cfg.model.pretrained)
        self.loss = U.build_loss(cfg.loss)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def criterion(self, outputs, labels, label_masks):
        pass

    def training_step(self, batch, batch_idx):
        inputs, labels, label_masks = batch

        outputs = self(inputs)
        loss, acc = self.criterion(outputs, labels, label_masks)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels, label_masks = batch

        outputs = self(inputs)
        loss, acc = self.criterion(outputs, labels, label_masks)

        self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=True)

        return acc

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, T_max=self.max_epochs
        # )
        scheduler_kwargs = dict(
            base_value=1.0,  # anneal from the original LR value,
            final_value=self.lr_cosine_min / self.lr,
            epochs=self.lr_cosine_epochs,
            warmup_start_value=self.lr_cosine_min / self.lr,
            warmup_epochs=self.lr_cosine_warmup_epochs,
            steps_per_epoch=self.lr_cosine_steps_per_epoch,
        )
        print("Cosine annealing with warmup restart")
        print(scheduler_kwargs)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer, lr_lambda=U.CosineScheduler(**scheduler_kwargs),
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


class BrainTumorTrainer:
    def __init__(self, cfg: DictConfig) -> None:
        OmegaConf.set_struct(cfg, False)
        self.cfg = cfg

        exp_name = self.generate_exp_name(cfg)
        exp_dir = f"{cfg.exp_root_dir}/{exp_name}"
        print("Exp name:", exp_name)
        self.exp_dir = U.f_expand(exp_dir)
        print("Exp dir:", self.exp_dir)

        U.f_mkdir(self.exp_dir)
        U.f_mkdir(U.f_join(self.exp_dir, "tb"))
        U.f_mkdir(U.f_join(self.exp_dir, "logs"))
        U.f_mkdir(U.f_join(self.exp_dir, "ckpt"))
        U.omegaconf_save(cfg, self.exp_dir, "conf.yaml")

        self.ckpt_cfg = cfg.checkpoint
        self.data_module = self.create_data_module(cfg)
        self.module = self.create_module(cfg)
        self.trainer = self.create_trainer(cfg.train)

    def create_data_module(self, cfg) -> pl.LightningDataModule:
        return BrainTumorDataModule(
            cfg=cfg.dataset,
            batch_size=cfg.train.batch_size,
            num_workers=cfg.train.num_workers,
        )

    def create_module(self, cfg) -> pl.LightningModule:
        # lr_cosine_steps_per_epoch = math.ceil(
        #     cfg.dataset.length / (cfg.train.devices * cfg.train.batch_size)
        # )
        return BrainTumor(cfg)

    def generate_exp_name(self, cfg):
        if cfg.exp_name is not None:
            return cfg.exp_name
        exp_name = cfg.model.type
        exp_name += f"_lr{cfg.train.lr:.0e}".replace("e-0", "e-")
        exp_name += f"_wd{cfg.train.weight_decay:.0e}".replace("e-0", "e-")
        exp_name += f"_lrmin{cfg.train.scheduler.lr_cosine_min:.0e}".replace(
            "e-0", "e-"
        )
        exp_name += f"_warmup{cfg.train.scheduler.lr_cosine_warmup_epochs}"
        exp_name += f"_maxep{cfg.train.max_epochs}"
        return exp_name

    def create_loggers(self):
        return [
            pl_loggers.TensorBoardLogger(self.exp_dir, name="tb", version=""),
            pl_loggers.CSVLogger(self.exp_dir, name="logs", version=""),
        ]

    def create_callbacks(self):
        ckpt = ModelCheckpoint(dirpath=U.f_join(self.exp_dir, "ckpt"), **self.ckpt_cfg)
        ckpt.FILE_EXTENSION = ".pth"
        return [ckpt, LearningRateMonitor("step")]

    def create_trainer(self, cfg) -> pl.Trainer:
        return pl.Trainer(
            max_epochs=cfg.max_epochs,
            accelerator="gpu",
            strategy=cfg.strategy,
            devices=cfg.devices,
            logger=self.create_loggers(),
            callbacks=self.create_callbacks(),
        )

    def fit(self):
        return self.trainer.fit(self.module, datamodule=self.data_module)

    # def test(self):
    #     return self.trainer.test(
    #         self.module, datamodule=self.data_module, ckpt_path=self.model.pretrained
    #     )
