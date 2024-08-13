import numpy as np

def padding(seqs, input_dim, maxlen):
    lengths = np.array([len(seq) for seq in seqs]).astype("int32")
    n_samples = len(seqs)

    x = np.zeros([n_samples, maxlen, input_dim]).astype(np.float32)
    for idx, seq in enumerate(seqs):
        for xvec, subseq in zip(x[idx, :, :], seq):
            xvec[subseq] = 1.
    return x, lengths
