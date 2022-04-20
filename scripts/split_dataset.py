"""
Script for spliting dataset into train and validation.

Author: Haoyi Zhu
"""

import os
import numpy as np
import json

root = "./data/classification/train"
exclude_ids = ["00109", "00123", "00709"]


def split(val_ratio=0.1, seed=42):
    np.random.seed(seed)

    all_img_ids = os.listdir(root)
    for exclude_id in exclude_ids:
        all_img_ids.remove(exclude_id)

    val_num = int(val_ratio * len(all_img_ids))
    val_ids = np.random.choice(all_img_ids, val_num, replace=False).tolist()

    with open(f"./train_val_splits/val_ids_seed{seed}_ratio{val_ratio}.json", "w") as f:
        json.dump(val_ids, f)

    return


if __name__ == "__main__":
    for val_ratio in [0.1, 0.2, 0.3, 0.4]:
        split(val_ratio=val_ratio)
