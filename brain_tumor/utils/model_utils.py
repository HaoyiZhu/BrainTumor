import torch
import brain_tumor.models as models
from brain_tumor.criterion import Criterion


def build_model(cfg):
    return models.names[cfg.type](**cfg.args)


def build_loss(cfg):
    return Criterion(cfg.task, **cfg.args)


def calc_accuracy(preds, labels):
    preds = preds.cpu().data.numpy()
    labels = labels.cpu().data.numpy()

    preds = preds.argmax(-1)

    return (preds == labels).mean()


def load_checkpoint(model, checkpoint, strict=True):
    if checkpoint is None or checkpoint == "":
        print(f"Train from scratch...")
    else:
        print(f"Loading checkpoint from {checkpoint}...")
        state_dict = torch.load(checkpoint, map_location=torch.device("cpu"))
        state_dict = _trim_state_dict(state_dict)
        model.load_state_dict(state_dict, strict=strict)
    # if it's a fp16 model, turn it back.
    if next(model.parameters()).dtype == torch.float16:
        model = model.float()
    return model


def _trim_state_dict(state_dict):
    from collections import OrderedDict

    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if "model" in state_dict:
        state_dict = state_dict["model"]
    ret_state_dict = OrderedDict()
    for (
        key,
        value,
    ) in state_dict.items():
        if key.startswith("model"):
            key = key[len("model.") :]
        ret_state_dict[key] = value
    return ret_state_dict
