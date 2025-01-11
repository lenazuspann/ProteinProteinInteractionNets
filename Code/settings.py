import os
import numpy as np

# data set and paths
path_data = os.path.join(os.path.abspath(os.getcwd()), 'data', 'dataset', 'PPT-Ohmnet_tissues-combined.edgelist')
path_joblib = os.path.join(os.path.abspath(os.getcwd()), 'data', 'joblib_data')
path_plots = os.path.join(os.path.abspath(os.getcwd()), 'data', 'plots')
save_data = True
load_data = True

# subset selection
tissue_list = ['pancreas', 'ovary', 'neuron']
tissue = 'pancreas'

# plot parameters
save_fig = False
cutoff = 20

# clustering algorithm parameters
save_clusters = True
load_clusters = True
stopping_criterion_girvan_newman = 20
p = np.linspace(0.01, 0.1, num=3)
k = np.linspace(10, 30, num=3)
beta = np.linspace(0.2, 0.4, num=3)
m = np.linspace(20,40,num=3)