# Brain Tumor Classification and Segmentation

[Brain Tumor AI Challenge (2021)](https://www.rsna.org/education/ai-resources-and-training/ai-image-challenge/brain-tumor-ai-challenge-2021)

Classification: [RSNA-MICCAI Brain Tumor Radiogenomic Classification](https://www.kaggle.com/competitions/rsna-miccai-brain-tumor-radiogenomic-classification/data)

Segmentation: [RSNA-ASNR-MICCAI Brain Tumor Segmentation (BraTS)](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1)

## Installation

```bash
# 1. Create a conda virtual enviornment.
conda create -n brain_tumor python=3.9 -y
conda create brain_tumor

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

## Training

TBD

## Validation

TBD