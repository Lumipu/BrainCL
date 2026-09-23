Store the dataset in HDF5 format in this directory and name the file sFCN.h5.

The HDF5 file should contain three root-level datasets:

fcn: shape (N, 200, 200), functional connectivity matrices
label: shape (N,), binary labels
site: shape (N,), acquisition site information

The three datasets must contain the same number of samples and be aligned sample by sample.