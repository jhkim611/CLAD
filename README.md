# Class Label-aware Graph Anomaly Detection

The official source code for "Class Label-aware Graph Anomaly Detection", accepted at CIKM 2023.

## Overview

Unsupervised GAD methods assume the lack of anomaly labels, i.e., whether a node is anomalous or not. 
One common observation we made from previous unsupervised methods is that they not only assume the absence of such anomaly labels, but also the \textit{absence of class labels} (the class a node belongs to used in a general node classification task). In this work, we study the utility of class labels for unsupervised GAD; in particular, how they enhance the detection of \textit{structural anomalies}. To this end, we propose a Class Label-aware Graph Anomaly Detection framework (CLAD) that utilizes a limited amount of labeled nodes to enhance the performance of unsupervised GAD. 
Extensive experiments on ten datasets  demonstrate the superior performance of CLAD in comparison to existing unsupervised GAD methods, even in the absence of ground-truth class label information.

## Datasets

Each 'dataset_name.mat' contains the following attributes:
* **adj** : the adjacency matrix of graph 'dataset_name', stored as a scipy.sparse.csc_matrix.
* **feat** : the attribute matrix of graph 'dataset_name', stored as a numpy array.
* **clabels** : the class label information of each node in graph 'dataset_name', stored as a numpy array. For graphs not containing any ground-truth labels, e.g., Automotive, an empty array will be returned.
* **alabels** : the anomaly label information of each node in graph 'dataset_name', stored as a numpy array(0: benign, 1: anomaly). This information is only used for evaluation.

## How to run CLAD

After unzipping 'dataset.zip', folder 'dataset' should be in the same directory as 'run.py' and 'utils.py'.

As an example,

```
python run.py --dataset citeseer --gtcl true --num_epoch 500 --nhid 16 --dropout 0.5 --alpha 0.2 --device cuda:1
```

would train a two-layer GCN model with latent dimension=16 and dropout=0.5 for 500 iterations. Then the final anomaly detection score with $\alpha=0.2$ is returned.
