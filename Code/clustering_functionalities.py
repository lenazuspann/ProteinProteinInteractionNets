import networkx as nx
import itertools as it
import joblib

from settings import *

### calculate communities based on the Girvan-Newman algorithm
def get_girvan_newman_clusters(G: nx.Graph, stop: int=stopping_criterion_girvan_newman, save_clusters: bool=True, load_clusters: bool=True, path: str=path_joblib):
    if load_clusters and os.path.exists(os.path.join(path, 'communities_girvan_newman.joblib')):
        comm_gn =  joblib.load(os.path.join(path, 'communities_girvan_newman.joblib'))[-1]
    else:
        comm = nx.community.girvan_newman(G)
        limited = it.takewhile(lambda c: len(c) <= stop, comm)
        comm_gn = []
        for communities in limited:
            comm_gn.append(tuple(sorted(c) for c in communities))
        if save_clusters:
            joblib.dump(comm_gn, os.path.join(path,'communities_girvan_newman.joblib'))
    return comm_gn


### calculate communities based on the label propagation algorithm
def get_label_propagation_clusters(G: nx.Graph, save_clusters: bool=True, load_clusters: bool=True, path: str=path_joblib):
    if load_clusters and os.path.exists(os.path.join(path, 'communities_label_propagation.joblib')):
        comm_lp =  joblib.load(os.path.join(path, 'communities_label_propagation.joblib'))
    else:
        comm_lp = list(nx.algorithms.community.label_propagation_communities(G))
        if save_clusters:
            joblib.dump(comm_lp, os.path.join(path, 'communities_label_propagation.joblib'))
    return comm_lp


### get communities based on the Louvain algorithm
def get_louvain_clusters(G: nx.Graph, save_clusters: bool=True, load_clusters: bool=True, path: str=path_joblib):
    if load_clusters and os.path.exists(os.path.join(path, 'communities_louvain.joblib')):
        comm_l =  joblib.load(os.path.join(path, 'communities_louvain.joblib'))
    else:
        comm_l = nx.algorithms.community.louvain_communities(G, weight=None)
        if save_clusters:
            joblib.dump(comm_l, os.path.join(path, 'communities_louvain.joblib'))
    return comm_l