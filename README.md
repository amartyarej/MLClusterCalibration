# MLClusterCalibration

## Running the code

The training and evaluation part (NOT FOR plotting) needs an environment setup provided in Dortmund E4. For using this (preferably in GPU enabled machine), use:
```
conda activate swbase
```

The large dataset can be obtained from `/ceph/groups/e4/users/arej/MLClusterCalibration/data/Akt4EMTopo.topo-cluster.root`. A skimming macro is used (`skim.C`) to skim the required branches along with some selection that removes a large number of cluster entries and stores the skimmed dataset. It can be obtained from `/ceph/groups/e4/users/arej/MLClusterCalibration/data/skimmed_full.root`

The output folder is assumed to be at `out`, that has to be created as:
```
mkdir out
```

The first code you need reads the ROOT file that contains the clusters info, transforms the input features and split the dataset into train, val, test.
The whole dataset is also stored, unchanged, in order to be able to plot the prediction at the end w.r.t. variables we do not train on.
```
python read_csv_and_make_plots.py
```


For the ML part, there is one code, depending on the argument, trains, retrains, tests.

```
python train.py --train   --outdir out
python train.py --retrain --outdir out
python train.py --test    --outdir out
```


## Running the plotting script

The output of the training/testing code is a bunch of numpy arrays. You can either create your own code to plot, or rely on this.

The plotting part needs an older setup available in CVMFS (via lxplus). For using it:
```
source setup.sh
```

Then run the plotting script:
```
python plotting.py
```
