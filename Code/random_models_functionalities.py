import networkx as nx
import pandas as pd 
import joblib

from settings import *


### construct a random network with the same degree distribution as the original one, rewire the links and compare some of their properties
def get_configuration_model_comparision(original_graph: nx.Graph, save_data: bool=True, load_data: bool=True, path=path_joblib):
    if load_data and os.path.exists(os.path.join(path, 'df_comp_configuraton_model.joblib')):
        df_comp = joblib.load(os.path.join(path, 'df_comp_configuraton_model.joblib'))
    else:
        # calculate the degree sequence of the original graph
        degree_sequence = [d for n, d in original_graph.degree()]
        
        # generate a random network with the same degree sequence using the configuration model
        configuration_model_graph = nx.configuration_model(degree_sequence)
        # convert it to simple graph
        configuration_model_graph = nx.Graph(configuration_model_graph)  
        configuration_model_graph.remove_edges_from(nx.selfloop_edges(configuration_model_graph))
        
        # Compare network properties
        properties_comparison = {
                "number of nodes": (original_graph.number_of_nodes(), configuration_model_graph.number_of_nodes()),
                "number of edges": (original_graph.number_of_edges(), configuration_model_graph.number_of_edges()),
                "average clustering coefficient": (nx.average_clustering(original_graph), nx.average_clustering(configuration_model_graph)),
                "average path length": (nx.average_shortest_path_length(original_graph), nx.average_shortest_path_length(configuration_model_graph))
            }
        df_comp = pd.DataFrame.from_dict(properties_comparison).T.rename(columns={0: 'original network', 1: 'rewired network'})

        if save_data:
            joblib.dump(df_comp, os.path.join(path, 'df_comp_configuraton_model.joblib'))
    return df_comp

### helper function for multiple simple calls below
def compute_metrics(G: nx.Graph):
    avg_clustering_coeff = nx.average_clustering(G)
    avg_path_length = nx.average_shortest_path_length(G)
    return [avg_clustering_coeff, avg_path_length]


### function to compare the original network with random graphs, namely: Erdos-Renyi and Watts-Strogatz with different parameters
def get_comp_with_random_nets(original_graph: nx.Graph, p: list, k: list, beta: list, save_data: bool=True, load_data: bool=True, path=path_joblib):
    if load_data and os.path.exists(os.path.join(path, 'df_sw.joblib')):
        df_sw = joblib.load(os.path.join(path, 'df_sw.joblib'))
    else:
        # make sure that the networks we compare ours to have the same number of nodes
        N = len(original_graph.nodes())
        
        # calculate the metrics for comparison across the grid of parameters specified for p, k and beta in the settings
        res = [compute_metrics(original_graph)] + [compute_metrics(nx.erdos_renyi_graph(N, pp))  for pp in p] + [compute_metrics(nx.watts_strogatz_graph(N, int(kk), bb)) for kk in k for bb in beta]
        df_sw = pd.DataFrame(res, columns=['Average Clustering Coefficient', 'Average Path Length'], index=pd.Index(['protein-protein interaction (scale-free, gamma=-1.36'] + [f'Erdos-Renyi with p={pp}' for pp in p] + [f'Watts-Strogatz with k={kk}, beta={bb}' for kk in k for bb in beta]))
        
        if save_data:
            joblib.dump(df_sw, os.path.join(path, 'df_sw.joblib'))
    return df_sw


### function to compare the original network with preferential attachment
def get_comp_with_ba(original_graph: nx.Graph, m: list, save_data: bool=True, load_data: bool=True, path: str=path_joblib):
    if load_data and os.path.exists(os.path.join(path, 'df_ba.joblib')):
        df_ba = joblib.load(os.path.join(path, 'df_ba.joblib'))
    else:
        # make sure that the networks we compare ours to have the same number of nodes
        N = len(original_graph.nodes())

        # calculate metrics for original graph and BA-model with parameters specified for m in settings
        res = [compute_metrics(original_graph)] + [compute_metrics(nx.barabasi_albert_graph(n=N, m=int(val))) for val in m]
        df_ba = pd.DataFrame(res, columns=['Average Clustering Coefficient', 'Average Path Length'], index=pd.Index(['protein-protein interaction (scale-free, gamma=-1.36'] + [f'Barabasi-Albert with m={val}' for val in m]))
        
        if save_data:
            joblib.dump(df_ba, os.path.join(path, 'df_ba.joblib'))
    return df_ba
