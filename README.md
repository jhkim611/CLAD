# LAAD

After unzipping 'dataset.zip', folder 'dataset' should be in the same directory as 'run.py' and 'utils.py'.

The LAAD framework can be run by the following command:

```
python run.py
```

with appropriate parameters.

For example,

```
python run.py --dataset citeseer --gtcl true --num_epoch 500 --nhid 16 --dropout 0.5 --alpha 0.2 --device cuda:1
```
