# DSMC-NN
The python script DSMC_datagen.py generates data points for distribution at f0, f1, ..., f50.
A sample of generated data which includes 100 points is also provided (data_hist.npz). One can read the file with a py script like:

import numpy as np;

loaded = np.load('data_hist.npz');

data = loaded['a'];

Note that here:
data[0:number of data points][0:number of time steps][0:number of grid points]
