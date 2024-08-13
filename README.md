# STEDR: A Deep Subgrouping Framework for Precision Drug Repurposing via Emulating Clinical Trials on Real-world Patient Data

## Introduction
This repository contains source code for the KDD 2025 submission paper "A Deep Subgrouping Framework for Precision Drug Repurposing via Emulating Clinical Trials on Real-world Patient Data".

Drug repurposing identifies new therapeutic uses for existing drugs, reducing the time and costs compared to traditional de novo drug discovery. Most existing drug repurposing studies using real-world patient data often treat the entire population as homogeneous, ignoring the heterogeneity of treatment responses across patient subgroups. This approach may overlook promising drugs that benefit specific subgroups but lack notable treatment effects across the entire population, potentially limiting the number of repurposable candidates identified. To address this, we introduce STEDR, a novel drug repurposing framework that integrates subgroup analysis with treatment effect estimation. Our approach first identifies repurposing candidates by emulating multiple clinical trials on real-world patient data and then characterizes patient subgroups by learning subgroup-specific treatment effects. We deploy STEDR to Alzheimer's Disease (AD), a condition with few approved drugs and known heterogeneity in treatment responses. We emulate trials for over one thousand medications on a large-scale real-world database covering over 8 million patients, identifying 14 drug candidates with beneficial effects to AD in characterized subgroups. Experiments demonstrate STEDR's superior capability in identifying repurposing candidates compared to existing approaches. Additionally, our method can characterize clinically relevant patient subgroups associated with important AD-related risk factors, paving the way for precision drug repurposing.


## Overview
![figure1](https://github.com/user-attachments/assets/649b103d-8159-49f6-b23c-ba5701ab4b8f)


Figure 1: An illustration of STEDR. The EHR data is processed through patient-level attention to learn individualized representations. The subgroup representation network assigns each subject to a subgroup and extracts subgroup-specific representations. The TEE model predicts the potential outcomes and propensity score from these subgroup-specific representations. The model is trained using the IPTW-based loss for confounder adjustment.


## Installation
Our model depends on Numpy, and PyTorch (CUDA toolkit if use GPU). You must have them installed before using our model
>
* Python 3.9
* Pytorch 1.10.2
* Numpy 1.21.2
* Pandas 1.4.1

This command will install all the required libraries and their specified versions.
```python 
pip install -r requirements.txt
```

## Data preparation
### Synthetic datasets
The downloadable version of the synthetic dataset used in the paper can be accessed in the 'data' folder. 

The structure of the synthetic data:
```
synthetic (dict)     
    |-- 'X': [...]   
    |-- 'T': [...]  
    |-- 'Y': [...]  
    |-- 'TE': [...]  
```
_Note: The simulation for the synthetic dataset is already integrated within 'train.py' file._


### OUD dataset
Please be informed that the real-world dataset utilized in this study is derived from MarketScan claims data. To obtain access to the data, interested parties are advised to contact IBM through [link](https://www.merative.com/real-world-evidence).

## Training and test
### Python command
For training and evaluating the model, run the following code
```python 
# Note 1: hyper-parameters are included in config.json.
# Note 2: the code simulates the data.
python train.py --config 'config.json' --data 'Synthetic'
```
  
### Parameters
Hyper-parameters are set in train.py
>
* `data`: dataset to use; {'Synthetic', 'IHDP'}.
* `config`: .json file

Hyper-parameters are set in *.json
>
* `n_samples`: the number of simulated samples (for the synthetic dataset only)
* `train_ratio`: the ratio of training
* `test_ratio`: the ratio of test set
* `maxlen`: the maximum number of visits (for the EHR dataset only)
* `n_groups`: the number of subgroups to identify.
* `att_dim`: the hidden dimension of the covariate-level and visit-level attentions.
* `emb_dim`: the hidden dimension of the transformer encoder.
* `dist_dim`: the hidden dimension of the local and global distributions and prediction networks.
* `n_layers`: the number of layers in TransformerEncoder
* `alpha`: weights to control the CI loss.
* `metrics`: metrics to print out. It is a list format. Functions for all metrics should be included in 'model/metric.py'.
* `early_stop`: the number of epochs for early stopping
* `monitor`: the criterion for early stopping. The first word is 'min' or 'max', and the second one is metric


_* Experiments were conducted using a computing cluster consisting of 42 nodes, each equipped with dual Intel Xeon 8268 processors, 384GB RAM, and dual NVIDIA Volta V100 GPUs with 32GB memory._






