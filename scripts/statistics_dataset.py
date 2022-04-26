"""
Statistics on
Brain Tumor Segmentation Dataset and
Brain Tumor Radiogenomic Classification Dataset.

Author: Haoxuan Sun
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import numpy as np
import seaborn as sns
from tqdm import tqdm
import brain_tumor.datasets.brat_classification_dataset as b_c_d
import brain_tumor.datasets.brat_segmentation_dataset as b_s_d


def analyse(ClassificationDataset, threshold=0):
    shape_0 = []
    means_1 = []
    means_2 = []
    for i in tqdm(range(len(dataset))):
        img, label = dataset.__getitem__(i)
        shape_0.append(img.shape[0])
        for j in range(img.shape[0]):
            mean1 = np.mean(img[j])
            means_1.append(mean1)
            if mean1 > threshold:  # 此处可升级
                means_2.append(mean1)
    all_mean = np.mean(means_1)
    threshold_mean = np.mean(means_2)

    vars_1 = []
    vars_2 = []

    for i in tqdm(range(len(dataset))):
        img, label = dataset.__getitem__(i)
        shape_0.append(img.shape[0])
        for j in range(img.shape[0]):
            var1 = np.var(img[j])
            vars_1.append(var1)
            if np.mean(img[j]) >= threshold:
                vars_2.append(var1)

    all_var = np.mean(vars_1)
    threshold_var = np.mean(vars_2)
    print(
        "shape_0:  max:"
        + str(np.max(shape_0))
        + "  min:"
        + str(np.min(shape_0))
        + "  mean:"
        + str(np.mean(shape_0))
        + "  std:"
        + str(np.std(shape_0))
    )
    print("all_mean=" + str(all_mean))
    print("threshold_mean=" + str(threshold_mean))
    print("all_std=" + str(np.sqrt(all_var)))
    print("threshold_std=" + str(np.sqrt(threshold_var)))
    sns.kdeplot(means_1)  # 密度图
    plt.show()


if __name__ == "__main__":
    
    sns.set()
    '''dataset = b_c_d.BraTClassificationDataset(
        root="H:\hw\BrainTumor\data\classification",
        mri_type=["T1wCE" ],  # "FLAIR", "T1w" , "T2w", "T1wCE"],
    )'''

    dataset = b_s_d.BraTSegmentationDataset(
        root='H:\hw\BrainTumor\data\segmentation',
        mri_type=["T1wCE"], # "FLAIR",'T1w', "T2w", "T1wCE"],
        )
    analyse(dataset)

