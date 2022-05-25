import torch
import torch.nn.functional as F
import brain_tumor.models as models
from brain_tumor.criterion import Criterion


def build_model(cfg):
    if cfg.dataset.img_dim == 2.5:
        num_classes = cfg.model.args.get("num_classes", 2)
        mlp_input_dim = cfg.model.get("mlp_input_dim", 256)
        cfg.model.args.update({"num_classes": mlp_input_dim})

        model = models.names[cfg.model.type](**cfg.model.args)

        norm_layer = None if cfg.train.batch_size == 1 else "batchnorm"
        cfg.model.mlp_args.update({"norm_layer": norm_layer})
        mlp_model = models.names["mlp"](**cfg.model.mlp_args)
        model.mlp_model = mlp_model

        return model
    else:
        return models.names[cfg.model.type](**cfg.model.args)


def build_loss(cfg):
    return Criterion(cfg.task, **cfg.args)


def calc_accuracy(preds, labels, average_outputs=False):
    if average_outputs:
        assert labels.shape[0] == 1, labels.shape
        preds = F.softmax(preds, dim=1)
        preds = preds.mean(0, keepdims=True)

    preds = preds.cpu().data.numpy()
    labels = labels.cpu().data.numpy()
    preds = preds.argmax(-1, keepdims=True)

    return (preds == labels).mean()


def load_checkpoint(model, checkpoint, strict=True):
    if checkpoint is None or checkpoint == "":
        print(f"Train from scratch...")
    else:
        print(f"Loading checkpoint from {checkpoint}...")
        state_dict = torch.load(checkpoint, map_location=torch.device("cpu"))
        state_dict = _trim_state_dict(state_dict)
        model.load_state_dict(state_dict, strict=strict)
    # # if it's a fp16 model, turn it back.
    # if next(model.parameters()).dtype == torch.float16:
    #     model = model.float()
    return model


def _trim_state_dict(state_dict):
    from collections import OrderedDict

    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if "model" in state_dict:
        state_dict = state_dict["model"]
    ret_state_dict = OrderedDict()
    for (key, value,) in state_dict.items():
        if key.startswith("model"):
            key = key[len("model.") :]
        ret_state_dict[key] = value
    return ret_state_dict
