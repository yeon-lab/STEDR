# STEDR: Subgroup-based Treatment Effect Estimation for Drug Repurposing

## Introduction
This repository contains source code for paper "STEDR: Subgroup-based Treatment Effect Estimation for Drug Repurposing".

Drug repurposing identifies new therapeutic uses for existing drugs, potentially reducing time and costs compared to developing new drugs. Existing studies for computational drug repurposing primarily focus on estimating the treatment effects of candidate drugs across the entire population and identifying the most beneficial ones for repurposable drugs. However, these approaches often overlook the heterogeneity in treatment responses among diverse populations. Addressing this variability is crucial for achieving precise drug repurposing, which aims to tailor therapies to individual patient needs, thereby enhancing treatment effect.
In this study, we introduce a novel framework, named STEDR, that integrates subgroup identification with treatment effect estimation, aimed at both advancing estimation and facilitating precise drug repurposing. STEDR effectively identifies drugs that can be repurposed, along with subgroups exhibiting heterogeneous treatment responses, by learning subgroup-specific treatment effects. Comprehensive experiments demonstrate the superior performance and efficiency of the proposed method. Notably, a real-world case study on Alzheimer's Disease (AD) treatment highlights the method's effectiveness in identifying suitable repurposable drug candidates and the specific subgroups that benefit most for precise drug repurposing.

## Overview
![figure1](https://github.com/yeon-lab/STEDR/assets/39074545/c9c5fc7b-bf14-4339-b5c0-9360c0081bd2)

Figure 1: An illustration of the proposed method. The input is first processed through the patient-level representation with covariate and visit attention mechanisms and a transformer encoder. Subgroup-specific representation learning assigns the subgroup and extracts subgroup-specific representation from its distribution based on similarities with the global distribution. To learn the subgroup-specific distributions, the model obtains a Gaussian mixture model using the subgroup-specific distributions and their similarities and then computes KLD between the mixture model and the global distribution. The model predicts the outcomes and propensity score from the subgroup-specific representation.


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
The downloadable version of the synthetic dataset used in the paper can be accesse in the 'data' folder. 

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
* `config`: json file

Hyper-parameters are set in *.json
>
* `n_samples`: the number of simulated samples (for the synthetic dataset only)
* `train_ratio`: the ratio of training set
* `test_ratio`: the ratio of test set
* `n_clusters`: the number of subgroups to identify.
* `att_dim`: the hidden dimension of the covariate-level and visit-level attentions.
* `emb_dim`: the hidden dimension of the transformer encoder.
* `dist_dim`: the hidden dimension of the local and global distributions and prediction networks.
* `n_layers`: the number of layers in TransformerEncoder
* `alpha`: weights to control the CI loss.
* `metrics`: metrics to print out. It is a list format. Functions for all metrics should be included in 'model/metric.py'.
* `early_stop`: the number of epochs for early stopping
* `monitor`: the criterion for early stopping. The first word is 'min' or 'max', the second one is metric


_* Experiments were conducted using a computing cluster consisting of 42 nodes, each equipped with dual Intel Xeon 8268 processors, 384GB RAM, and dual NVIDIA Volta V100 GPUs with 32GB memory._






