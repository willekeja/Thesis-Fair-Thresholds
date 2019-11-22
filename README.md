# Thesis-Fair-Thresholds

This repository contains all files to run the models from the thesis. Examples of how to run models are contained in the run folder. To run them they need to be moved to the same directory as the other files, however. Each run saves the statistics and predicted values. Saved predictions from runs for the thesis cannot be provided on github as they far exceed the size limit.

The file names are mostly descriptive, the following provides an additional summary:

1) replication_results_vfae.ipynb and replication_results_adversarial_learning.ipynb provide replications of the results reported in the original papers
2) lipton.ipynb provides a comparison to the post-processing proposed by lipton et. al.
3) algorithms are defined in adversarial.py (adversarial models), vfae.py (variational fair autoencoder), pp_nn.py (post-processing for neural network) and algorithms_new (remaining models)
4) utils_new.py contains several functions used throughout the remaining scripts
5) datasets are compas.csv, credit.csv, residencia.xlsx and Data.dat (adult)
