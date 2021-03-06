# AI3601 Final Project: Brain Tumor Radiogenomic Classification

[Brain Tumor AI Challenge (2021)](https://www.rsna.org/education/ai-resources-and-training/ai-image-challenge/brain-tumor-ai-challenge-2021)

Kaggle: [RSNA-MICCAI Brain Tumor Radiogenomic Classification](https://www.kaggle.com/competitions/rsna-miccai-brain-tumor-radiogenomic-classification/data)

Author: [Haoyi Zhu](https://github.com/HaoyiZhu/), [Junjie Huang](https://github.com/Jessica-legend), [Haoxuan Sun](https://github.com/guwangtu)

## Installation

```bash
# 1. Create a conda virtual enviornment.
conda create -n brain_tumor python=3.9 -y
conda activate brain_tumor

# 2. Install PyTorch >= 1.10 according to your CUDA version. For example:
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

# 3. Get the repo
git clone https://github.com/HaoyiZhu/BrainTumor.git
cd BrainTumor

# 4. Install
pip install -e .
```

## Prepare dataset

First get [kaggle API](https://github.com/Kaggle/kaggle-api) by running `pip install kaggle`, then you may need to register your kaggle account following [here](https://blog.csdn.net/qq_40263477/article/details/107801843). Next download brain tumor segmentation and classification datasets:

```bash
kaggle competitions download -c rsna-miccai-brain-tumor-radiogenomic-classification
kaggle datasets download -d dschettler8845/brats-2021-task1
```

Finally, extract the datasets to `./data` and make them look like this:

```
|-- brain_tumor
|-- configs
|-- scripts
|-- train_val_splits
|-- data
`-- |-- classification
    |   |-- sample_submission.csv
    |	|-- train_labels.csv
    |   |-- test
    |   |   |-- 00001
    |   |   |-- 00013
    |   |   |-- ... 
    |   `-- train
    |   	|-- 00000
    |       |-- 00002
    |       |-- ... 
    `-- segmentation
    	|-- BraTS2021_00495
    	|	|-- BraTS2021_00495_flair.nii.gz
    	|	|-- ...
    	|-- BraTS2021_00621
    	|	|-- BraTS2021_00621_flair.nii.gz
    	|	|-- ...
    	`-- BraTS2021_Training_Data
    	 	|-- BraTS2021_00000
    	 	|	|-- BraTS2021_00000_flair.nii.gz
    	 	|	|-- ...
    		|-- BraTS2021_00002
    	 	|	|-- BraTS2021_00002_flair.nii.gz
    	 	|	|-- ...
    	 	|-- ...
	
	
```

## ANN Training

```bash
# Choose one or multiple gpus
# Number of gpus should match with ${train.devices} in config
# You can simply overwrite the parameters in config by xxx=xxx in command
CUDA_VISIBLE_DEVICES=0,1 python scripts/train.py train.devices=2
```

## SNN Training

```bash
# -e means the experiment specification file you choose
# -g means the gpu id you want to use
python scripts/snn_train.py -e configs/snn.yaml -g 0
python scripts/snn_train.py -e configs/snn_new.yaml -g 1
```