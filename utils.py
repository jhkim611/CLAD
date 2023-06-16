import numpy as np
import torch
import torch.nn as nn

import random

from sklearn.cluster import KMeans

from sklearn.preprocessing import minmax_scale
scaler = minmax_scale

import scipy.io as sio
import math

def load_dataset(dataset, gtcl):
    data = sio.loadmat("./dataset/{}.mat".format(dataset))

    adj = data['adj'].astype(np.int64)
    feat = data['feat'].astype(np.float32)
    
    if gtcl=='true':
        clabels = data['clabels'][0].astype(np.int64)
        idx_all = list(range(adj.shape[0]))
        random.shuffle(idx_all)
        idx_known = idx_all[:int(adj.shape[0]*0.3)]
    elif gtcl=='false':
        print('Assigning pseudo-labels...')
        k = 5
        model = KMeans(n_clusters=k)
        model.fit(feat)
        clabels = model.labels_.astype(np.int64)

        nodes = [[i for i in range(len(clabels)) if clabels[i]==j] for j in range(k)]
        distances = [[np.linalg.norm(feat[i] - model.cluster_centers_[clabels[i]]) for i in range(len(clabels)) if clabels[i]==j] for j in range(k)]
        idx_known = np.hstack(np.array([[np.array(nodes[i])[np.array(distances[i]).argsort()][:min(len(distances[i]), 50)]] for i in range(k)], dtype=object).reshape(-1))

    alabels = data['alabels'][0].astype(np.int64)

    return adj, feat, clabels, alabels, idx_known
    
def shannon_entropy(array):
      return np.sum(-array*np.log(array))

def JSDplus(_output, _A, i):
    neighbors = np.nonzero(_A[i])[0].tolist()
    N = len(neighbors)

    s = 0
    for n in range(N):
        if n==0:
            s = shannon_entropy(_output[neighbors[n]])
        else:
            s += shannon_entropy(_output[neighbors[n]])
            
    s += shannon_entropy(_output[i])
    
    gamma = sum([np.argmax(_output[neighbors[n]])==np.argmax(_output[i]) for n in range(N)])

    if N==0:
        kappa = math.log(1)
    elif gamma<N:
        kappa = math.log(N - gamma)
    else:
        kappa = math.log(1)

    return (shannon_entropy(np.mean(_output[neighbors+[i]], axis=0)) - s/(N+1))*kappa

def ED(_feat, _A, i):
    neighbors = np.nonzero(_A[i])[0].tolist()
    N = len(neighbors)
    
    ed_sum = 0
    for n in range(N):
        ed_sum += np.linalg.norm(_feat[neighbors[n]] - _feat[i])
    
    if N==0:
        N += 1
    return ed_sum/N