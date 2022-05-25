import hydra
import yaml
from omegaconf import OmegaConf

from brain_tumor.utils.dataloader import loading_data
from brain_tumor.utils.launcher_utils import generate_exp_name
import brain_tumor.utils.pytorch_util as ptu
from brain_tumor.utils import logger
from brain_tumor.utils.launcher_utils import setup_logger
from brain_tumor.learn.snn_trainer import snn_train
import brain_tumor.utils as U


@hydra.main(config_path="../configs", config_name="snn")
def main(cfg):
    if cfg.use_gpu:
        device = ptu.set_gpu_mode(True, cfg.gpu)

    # Set the random seed manually for reproducibility.
    seed = cfg.seed
    ptu.set_seed(seed)

    with open(cfg.experiment, "r") as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.load(spec_string, Loader=yaml.Loader)

    OmegaConf.set_struct(cfg, False)

    exp_name = generate_exp_name(cfg)
    exp_dir = f"{cfg.exp_root_dir}/{exp_name}"
    print("Exp name:", exp_name)
    exp_dir = U.f_expand(exp_dir)
    print("Exp dir:", exp_dir)

    U.f_mkdir(exp_dir)
    U.f_mkdir(U.f_join(exp_dir, "tb"))
    U.f_mkdir(U.f_join(exp_dir, "logs"))
    U.f_mkdir(U.f_join(exp_dir, "ckpt"))
    U.omegaconf_save(cfg, exp_dir, "conf.yaml")

    train_data_loader, val_data_loader, test_data_loader = loading_data(
        cfg=cfg,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
    )
    dataloader = [train_data_loader, val_data_loader, test_data_loader]

    setup_logger(log_dir=cfg.exp_root_dir, variant=exp_specs)
    snn_train(dataloader, cfg, logger, device)


if __name__ == "__main__":
    main()
