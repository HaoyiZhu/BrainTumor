from __future__ import annotations

from omegaconf import DictConfig
from brain_tumor.utils import logger

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils

import brain_tumor.utils as U


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
        for data, targets, _ in train_loader:
            data = data.to(device)
            targets = targets.to(device)
            spk_rec, _ = forward_pass(net, num_steps, data)

            acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
            total += spk_rec.size(1)

    return acc / total


def snn_train(dataloader, cfg: DictConfig, device, lr_cosine_steps_per_epoch: int = 1):
    lr = cfg.train.lr
    max_epochs = cfg.train.max_epochs
    weight_decay = cfg.train.weight_decay
    lr_cosine_min = cfg.train.scheduler.lr_cosine_min
    lr_cosine_epochs = cfg.train.scheduler.lr_cosine_epochs
    lr_cosine_warmup_epochs = cfg.train.scheduler.lr_cosine_warmup_epochs

    train_data_loader, val_data_loader, test_data_loader = (
        dataloader[0],
        dataloader[1],
        dataloader[2],
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
        nn.Linear(64 * 29 * 29, 2),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True),
    ).to(device)
    loss_fn = SF.ce_rate_loss()
    opt = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999)
    )
    scheduler_kwargs = dict(
        base_value=1.0,  # anneal from the original LR value,
        final_value=lr_cosine_min / lr,
        epochs=lr_cosine_epochs,
        warmup_start_value=lr_cosine_min / lr,
        warmup_epochs=lr_cosine_warmup_epochs,
        steps_per_epoch=lr_cosine_steps_per_epoch,
    )
    print("Cosine annealing with warmup restart")
    print(scheduler_kwargs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=opt, lr_lambda=U.CosineScheduler(**scheduler_kwargs),
    )
    best_val_loss = None
    for current_epoch in range(max_epochs):
        model.train()
        for i, batches in enumerate(train_data_loader):
            opt.zero_grad()

            inputs, labels, _ = batches
            labels = labels.squeeze()
            inputs = inputs.to(device)
            labels = labels.to(device)

            spk_rec, mem_rec = forward_pass(model, cfg.train.num_steps, inputs)

            loss = loss_fn(spk_rec, labels)
            loss.backward()
            opt.step()

        acc = batch_accuracy(train_data_loader, model, cfg.train.num_steps, device)
        logger.record_tabular("train_loss", loss.cpu().data.numpy())
        logger.record_tabular("train_acc", acc.cpu().data.numpy())

        model.eval()
        with torch.no_grad():
            for i, batches in enumerate(val_data_loader):
                inputs, labels, _ = batches
                labels = labels.squeeze()
                inputs = inputs.to(device)
                labels = labels.to(device)

                spk_rec = forward_pass(model, cfg.train.num_steps, inputs)
                val_loss = loss_fn(spk_rec, labels)
            val_acc = batch_accuracy(
                train_data_loader, model, cfg.train.num_steps, device
            )
            logger.record_tabular("val_acc", val_acc.cpu().data.numpy())
            logger.record_tabular("epoch", current_epoch)
            logger.dump_tabular(with_prefix=False, with_timestamp=False)
            scheduler.step()
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                logger.save_torch_model(model, "model.pt")
                best_val_loss = val_loss
