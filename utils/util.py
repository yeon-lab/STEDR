import json
from pathlib import Path
from collections import OrderedDict
from itertools import repeat
import pandas as pd
import numpy as np

def padding(seqs, input_dim, maxlen):
    lengths = np.array([len(seq) for seq in seqs]).astype("int32")
    n_samples = len(seqs)
    x = np.zeros([n_samples, maxlen, input_dim]).astype(np.float32)
    for idx, seq in enumerate(seqs):
        for xvec, subseq in zip(x[idx, :, :], seq):
            xvec[subseq] = 1.
    return x, lengths
    
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
