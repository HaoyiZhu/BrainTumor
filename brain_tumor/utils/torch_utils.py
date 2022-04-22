import torch
import numpy as np
import math


def set_seed(seed=43211):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def generate_cosine_schedule(
    base_value,
    final_value,
    epochs,
    steps_per_epoch,
    warmup_epochs=0,
    warmup_start_value=0,
) -> np.ndarray:
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * steps_per_epoch
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(warmup_start_value, base_value, warmup_iters)

    iters = np.arange(epochs * steps_per_epoch - warmup_iters)
    schedule = np.array(
        [
            final_value
            + 0.5
            * (base_value - final_value)
            * (1 + math.cos(math.pi * i / (len(iters))))
            for i in iters
        ]
    )
    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * steps_per_epoch
    return schedule


class CosineScheduler:
    def __init__(
        self,
        base_value,
        final_value,
        epochs,
        steps_per_epoch,
        warmup_epochs=0,
        warmup_start_value=0,
    ):
        """
        Args:
            epochs: effective epochs for the cosine schedule, *including* warmup
                after these epochs, scheduler will output `final_value` ever after
        """
        assert warmup_epochs < epochs
        self._effective_steps = epochs * steps_per_epoch
        self.schedule = generate_cosine_schedule(
            base_value=base_value,
            final_value=final_value,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            warmup_epochs=warmup_epochs,
            warmup_start_value=warmup_start_value,
        )
        assert self.schedule.shape == (epochs * steps_per_epoch,)
        self._final_value = final_value
        self._steps_tensor = torch.tensor(0, dtype=torch.long)  # for register buffer

    def register_buffer(self, module: torch.nn.Module, name="cosine_steps"):
        module.register_buffer(name, self._steps_tensor, persistent=True)

    def __call__(self, step):
        self._steps_tensor.copy_(torch.tensor(step))
        if step >= self._effective_steps:
            val = self._final_value
        else:
            val = self.schedule[step]
        return val


def set_requires_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad


def freeze_params(model):
    set_requires_grad(model, False)


def unfreeze_params(model):
    set_requires_grad(model, True)
