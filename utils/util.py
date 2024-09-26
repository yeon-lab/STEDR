import json
from pathlib import Path
from collections import OrderedDict
from itertools import repeat
import pandas as pd
import numpy as np
import pickle
import os
import random

SEED = 1111


def padding(seqs, input_dim, maxlen):
    lengths = np.array([len(seq) for seq in seqs]).astype("int32")
    n_samples = len(seqs)

    x = np.zeros([n_samples, maxlen, input_dim]).astype(np.float32)
    for idx, seq in enumerate(seqs):
        for xvec, subseq in zip(x[idx, :, :], seq):
            xvec[subseq] = 1.
    return x, lengths

def load_synthetic(params, is_importance=False):
    np.random.seed(SEED) 
    random.seed(SEED) 
    x_lengths = None
    if params["dataset"] == 'IHDP':
        data, BetaB = create_IHDP()
        n_samples = len(data[0])
    elif params["dataset"] == 'synthetic':
        n_samples = params["n_samples"]
        data, BetaB = create_synth(n_samples)
    else:
        if os.path.isfile('timeseries.pkl'):
            print('load existing timeseries data..')
            pkl_data = pickle.load(open('timeseries.pkl','rb'))
            data, x_lengths = pkl_data['data'], pkl_data['x_lengths']
            n_samples, _, n_feat = data[0].shape 
        else:
            print('save timeseries data..')
            model = AutoregressiveSimulation(x_noise=0.1, y_noise=0.1)
            data, x_lengths = model.generate_dataset(params["n_samples"], max_timesteps=20)
            BetaB = model.BetaB
            n_samples, _, n_feat = data[0].shape 
            pkl_dict = {}
            pkl_dict['data'] = data
            pkl_dict['x_lengths'] = x_lengths
            pkl_dict['y_coeffs'] = BetaB
            pkl_dict['x_coeffs'] = model.covariates_coefficients
            pickle.dump(pkl_dict, open('timeseries.pkl','wb'))
            
    n_train = int(n_samples*params["train_ratio"])
    n_test = int(n_samples*params["test_ratio"])

    index = np.random.RandomState(seed=SEED).permutation(n_samples)
    train_index = index[:n_train]
    test_index = index[n_train:n_train+n_test]
    valid_index = index[n_train+n_test:]

    train_set = split_dataset(data, train_index, x_lengths)
    valid_set = split_dataset(data, valid_index, x_lengths)
    test_set = split_dataset(data, test_index, x_lengths)
    
    if is_importance:
        return train_set, valid_set, test_set, BetaB
    
    return train_set, valid_set, test_set


def split_dataset(data, index, x_lengths):
    (X, T, Y, TE) = data
    dataset = {
        'X': X[index],
        'T': T[index],
        'Y': Y[index],
        'TE': TE[index],
        'x_lengths': None
    }
    if x_lengths is not None:
        dataset['x_lengths'] = x_lengths[index]
    return dataset



def create_synth(n_samples):
    np.random.seed(SEED)
    random.seed(SEED)
    X = np.round(np.random.normal(size=(n_samples, 1), loc=66.0, scale=4.1))  # age
    X = np.block([X, np.round(
        np.random.normal(size=(n_samples, 1), loc=6.2, scale=1.0) * 10.0) / 10.0])  # white blood cell count
    X = np.block(
        [X, np.round(np.random.normal(size=(n_samples, 1), loc=0.8, scale=0.1) * 10.0) / 10.0])  # Lymphocyte count
    X = np.block([X, np.round(np.random.normal(size=(n_samples, 1), loc=183.0, scale=20.4))])  # Platelet count
    X = np.block([X, np.round(np.random.normal(size=(n_samples, 1), loc=68.0, scale=6.6))])  # Serum creatinine
    X = np.block(
        [X, np.round(np.random.normal(size=(n_samples, 1), loc=31.0, scale=5.1))])  # Aspartete aminotransferase
    X = np.block([X, np.round(np.random.normal(size=(n_samples, 1), loc=26.0, scale=5.1))])  # Alanine aminotransferase
    X = np.block([X, np.round(np.random.normal(size=(n_samples, 1), loc=339.0, scale=51))])  # Lactate dehydrogenase
    X = np.block([X, np.round(np.random.normal(size=(n_samples, 1), loc=76.0, scale=21))])  # Creatine kinase
    X = np.block([X, np.floor(np.random.uniform(size=(n_samples, 1)) * 11) + 4])  # Time from study 4~14
    TIME = X[:, 9]

    X_ = pd.DataFrame(X)
    X_ = normalize_mean(X_)
    X = np.array(X_)

    T = np.random.binomial(1, 0.5, size=(n_samples,1))

    # sample random coefficients
    coeffs_ = [0, 0.1, 0.2, 0.3, 0.4]
    BetaB = np.random.choice(coeffs_, size=9, replace=True, p=[0.6, 0.1, 0.1, 0.1, 0.1])

    MU_0 = np.dot(X[:, 0:9], BetaB)
    MU_1 = np.dot(X[:, 0:9], BetaB)

    logi0 = lambda x: 1 / (1 + np.exp(-(x - 9))) + 5
    logi1 = lambda x: 5 / (1 + np.exp(-(x - 9)))

    MU_0 = MU_0 + logi0(TIME)
    MU_1 = MU_1 + logi1(TIME)

    Y_0 = (np.random.normal(scale=0.1, size=len(X)) + MU_0).reshape(-1,1)
    Y_1 = (np.random.normal(scale=0.1, size=len(X)) + MU_1).reshape(-1,1)

    Y = T * Y_1 + (1 - T) * Y_0
    Y_cf = T * Y_0 + (1 - T) * Y_1
    
    TE = Y_1 - Y_0

    return (X, T, Y, TE), BetaB


def create_IHDP(noise=0.1):
    np.random.seed(SEED)
    random.seed(SEED)
    Dataset= pd.read_csv("https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_1.csv", header = None)

    col = ["Treatment", "Response", "Y_CF", "mu0", "mu1", ]

    for i in range(1, 26):
        col.append("X" + str(i))
    Dataset.columns = col
    Dataset.head()

    num_samples = len(Dataset)

    feat_name = 'X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15 X16 X17 X18 X19 X20 X21 X22 X23 X24 X25'

    X = np.array(Dataset[feat_name.split()])
    T = np.array(Dataset['Treatment']).reshape(-1,1)
    
    Y_0 = np.array(np.random.normal(scale=noise, size=num_samples) + Dataset['mu0']).reshape(-1,1)
    Y_1 = np.array(np.random.normal(scale=noise, size=num_samples) + Dataset['mu1']).reshape(-1,1)
    BetaB = None    
    # ##################
    
    # coeffs_ = [0, 0.1, 0.2, 0.3, 0.4]

    # BetaB = np.random.choice(coeffs_, size=X.shape[1], replace=True, p=[0.6, 0.1, 0.1, 0.1, 0.1])
    
    # W = np.full_like(X, 0.5)
    # omega_sB = np.full(len(X), 4)
    
    # MU_0 = np.exp(np.dot(X+W, BetaB))
    # MU_1 = np.dot(X, BetaB) - omega_sB

    # Y_0 = (np.random.normal(scale=noise, size=len(X)) + MU_0).reshape(-1,1)
    # Y_1 = (np.random.normal(scale=noise, size=len(X)) + MU_1).reshape(-1,1)
    # ##################


    Y = T * Y_1 + (1 - T) * Y_0
    Y_cf = T * Y_0 + (1 - T) * Y_1

    TE = Y_1 - Y_0
    
    return (X, T, Y, TE), BetaB


def normalize_mean(df):
    result = df.copy()
    for feature_name in df.columns:
        result[feature_name] = (result[feature_name] - result[feature_name].mean()) / result[feature_name].std()
    return result



def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader
        
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('true', 't'):
        return True
    elif v.lower() in ('false', 'f'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
