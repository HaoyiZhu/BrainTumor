from __future__ import annotations
import yaml
import argparse
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils
from snntorch import backprop

from brain_tumor.utils.dataloader import loading_data
import brain_tumor.utils.pytorch_util as ptu
from brain_tumor.utils import logger
from brain_tumor.utils.launcher_util import setup_logger


def forward_pass(net, num_steps, data):
    mem_rec = []
    spk_rec = []
    utils.reset(net)  # resets hidden states for all LIF neurons in net

    for step in range(num_steps):
        spk_out, mem_out = net(data)
        spk_rec.append(spk_out)
        mem_rec.append(mem_out)

    return torch.stack(spk_rec), torch.stack(mem_rec)


def batch_accuracy(train_loader, net, num_steps, device):
    with torch.no_grad():
        total = 0
        acc = 0
        net.eval()

        train_loader = iter(train_loader)
        for data, targets in train_loader:
            data = data.to(device)
            targets = targets.to(device)
            spk_rec, _ = forward_pass(net, num_steps, data)

            acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
            total += spk_rec.size(1)

    return acc / total


def experiment(exp_specs, device):
    lr = exp_specs['lr']
    max_epochs = exp_specs['max_epochs']

    train_data_loader, val_data_loader, test_data_loader = loading_data(
        exp_specs=exp_specs, batch_size=exp_specs['batch_size'], num_workers=exp_specs['num_workers'],
    )
    spike_grad = surrogate.fast_sigmoid(slope=25)
    beta = 0.5
    model = nn.Sequential(
        nn.Conv2d(1, 12, 5),
        nn.MaxPool2d(2),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
        nn.Conv2d(12, 64, 5),
        nn.MaxPool2d(2),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
        nn.Flatten(),
        nn.Linear(64 * 6 * 6, 2),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True),
    ).to(device)
    loss_fn = SF.ce_rate_loss()
    opt = torch.optim.Adam(
        model.parameters(), lr=lr, betas=(0.9, 0.999)
    )

    for current_epoch in range(max_epochs):
        model.train()

        loss = backprop.BPTT(model, train_data_loader, optimizer=opt, criterion=loss_fn, 
            num_steps=exp_specs['num_steps'], time_var=False, device=device)

        # acc = batch_accuracy(train_data_loader, model, cfg.train.num_steps, device)
        logger.record_tabular("train_loss", loss.cpu().data.numpy().item())
        # logger.record_tabular("train_acc", acc)

        # model.eval()
        # with torch.no_grad():
        #     for i, batches in enumerate(val_data_loader):
        #         inputs, labels = batches
        #     #     labels = labels.squeeze()
        #         inputs = inputs.to(device)
        #         labels = labels.to(device)

        #         spk_rec, mem_rec = forward_pass(model, cfg.train.val_num_steps, inputs)
        #         val_loss = loss_fn(spk_rec, labels)
        #     # val_loss = backprop.BPTT(model, val_data_loader, optimizer=opt, criterion=loss_fn, 
        #     #     num_steps=cfg.train.val_num_steps, time_var=False, device=device)
        #     val_acc = batch_accuracy(
        #         val_data_loader, model, cfg.train.val_num_steps, device
        #     )
        # scheduler.step()
        # logger.record_tabular("val_loss", val_loss.cpu().data.numpy().item())
        # logger.record_tabular("val_acc", val_acc)
        logger.record_tabular("Epoch", current_epoch)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", help="experiment specification file")
    parser.add_argument("-g", "--gpu", help="gpu id", type=int, default=0)
    args = parser.parse_args()

    with open(args.experiment, "r") as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.load(spec_string, Loader=yaml.Loader)

    if exp_specs["use_gpu"]:
        device = ptu.set_gpu_mode(True, args.gpu)

    # Set the random seed manually for reproducibility.
    seed = exp_specs["seed"]
    ptu.set_seed(seed)

    setup_logger(log_dir=exp_specs["log_dir"], variant=exp_specs)
    experiment(exp_specs, device)
