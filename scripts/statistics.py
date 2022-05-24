import brain_tumor.datasets.brat_classification_dataset as cc
import brain_tumor.datasets.brat_segmentation_dataset as ss
import numpy as np
import os
import json
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset

from tqdm import tqdm

dataset = cc.BraTClassificationDataset(
    cfg=DictConfig({"split": {"root": "./train_val_splits", "seed": 42, "ratio": 0.1}}),
    root="H:\hw\BrainTumor\data\classification",
    mri_type=["FLAIR"],  # , "T2w", "T1wCE"],
)
i1, l1 = dataset.__getitem__(0)
min = np.min(i1)
max = np.max(i1)
for i in tqdm(range(len(dataset))):
    img, label = dataset.__getitem__(i)
    min = np.min([np.min(img), min])
    max = np.max([np.max(img), max])

print(min, max)
