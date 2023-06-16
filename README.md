# Label-Aware Graph Anomaly Detection

The official source code for Label-Aware Graph Anomaly Detection.

## Abstract

Graph anomaly detection (GAD) methods aim to find nodes that
show devious patterns in comparison to other nodes, which are
categorized into structural and attribute anomalies. In particular,
unsupervised GAD methods assume the lack of anomaly labels,
i.e., whether a node is anomalous or not. One common observation
we made from previous unsupervised methods is that they not
only assume the absence of anomaly labels, but also the absence
of class labels (the class a node belongs to used in a general node
classification task). In this work, we study the utility of class labels
for unsupervised GAD, i.e., we study how the class label informa-
tion of nodes enhances the detection of structural anomalies in
particular. To this end, we propose a Label-Aware Graph Anomaly
Detection framework (LAAD) that utilizes a limited amount of
labeled nodes to enhance the performance of unsupervised GAD.
Extensive experiments on ten datasets, including both synthetic
and real-world datasets, demonstrate the superior performance of
LAAD in comparison to existing unsupervised GAD methods, even
in the absence of ground-truth class label information.

## Overall Framework

![architecture](https://github.com/jhkim611/LAAD/assets/86581545/19ada2db-64af-4583-8e10-55bfaec04f6e)

## How to run LAAD

After unzipping 'dataset.zip', folder 'dataset' should be in the same directory as 'run.py' and 'utils.py'.

As an example,

```
python run.py --dataset citeseer --gtcl true --num_epoch 500 --nhid 16 --dropout 0.5 --alpha 0.2 --device cuda:1
```

would train a two-layer GCN model with latent dimension=16 and dropout=0.5 for 500 iterations. Then the final anomaly detection score with $\alpha=0.2$ is returned.
