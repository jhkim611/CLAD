import numpy as np
import torch

from utils import *
from deeprobust.graph.defense import GCN

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import minmax_scale
scaler = minmax_scale

import argparse

parser = argparse.ArgumentParser(description='LAAD: Label-Aware Graph Anomaly Detection')
parser.add_argument('--dataset', type=str, default='citeseer')  # 'cora' 'citeseer' 'amazon-computers' 'amazon-photo' 'ogbn-arxiv' 'Automotive' 'Pation_Lawn_and_Garden' 'Office_Products' 'Yelp' 'Elliptic'
parser.add_argument('--gtcl', type=str, default='true') # whether or not ground-truth class labels are available; true/false
parser.add_argument('--num_epoch', type=int, default=500) # # of epochs for GCN
parser.add_argument('--nhid', type=int, default=16) # latent dimension for GCN
parser.add_argument('--dropout', type=float, default=0.5) # dropout rate for GCN
parser.add_argument('--alpha', type=float, default=0.5) # balancing parameter
parser.add_argument('--device', type=str, default='cuda:1') # device; cpu or cuda

args = parser.parse_args()

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

print('Loading dataset...')
adj, feat, clabels, alabels, idx_known = load_dataset(args.dataset, args.gtcl)
print('Done!\n')

idx_train, idx_val = train_test_split(idx_known, test_size=0.05, random_state=42)

gcn = GCN(nfeat=feat.shape[1],
          nhid=args.nhid,
          nclass=clabels.max().item() + 1,
          dropout=args.dropout, device=device)
gcn.to(device)

gcn.fit(feat, adj, clabels, idx_train, idx_val, verbose=True, train_iters=500)

output = gcn.output.cpu().detach().numpy()
output = np.exp(output)
A = np.asarray(adj.todense())
F = feat

print('')
print('Obtaining structural anomaly scores...')
jsdplus_all = [JSDplus(output+1e-6, A, i) for i in range(len(alabels))]
jsdplus_scaled = scaler(torch.clamp(torch.tensor(jsdplus_all), max=np.quantile(jsdplus_all, 0.99)).numpy())

print('Obtaining attribute anomaly scores...')
ed_all = [ED(F, A, i) for i in range(len(alabels))]
ed_scaled = scaler(torch.clamp(torch.tensor(ed_all), max=np.quantile(ed_all, 0.99)).numpy())
print('Done!\n')

print('Final anomaly detection')
final_scores = args.alpha*np.array(jsdplus_scaled) + (1 - args.alpha)*np.array(ed_scaled)
print('*****************')
print('AUC score: {:.1f}'.format(100*roc_auc_score(alabels, final_scores)))
print('*****************')