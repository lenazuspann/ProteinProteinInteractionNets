import os
import joblib
import pandas as pd
import networkx as nx

from settings import *


### load the data set from the downloaded edge list
def load_dataset(path: str=path_data):
    return pd.read_csv(path, sep='\t', header=0).rename(columns={'# protein1': 'protein1'})


### calculate overview statistics
def get_overview_stats(df: pd.DataFrame, load_data: bool=True, save_data: bool=True, path: str=path_joblib):
    # load data set if possible
    if load_data and os.path.exists(os.path.join(path, 'df_stats.joblib')):
        df_stats = joblib.load(os.path.join(path, 'df_stats.joblib'))
    else:
        # delete the duplicate connections (same protein-protein interaction in different tissues)
        df_summary_stats = df[~df.filter(like='protein').apply(frozenset, axis=1).duplicated()].reset_index(drop=True)
        n_nodes = len(list(set(df_summary_stats['protein1'].values) | set(df_summary_stats['protein2'].values)))
        n_edges = df_summary_stats.shape[0]

        # join total number of edges and nodes with a groupby-object based on the tissue
        df_stats = pd.concat([pd.DataFrame(['total', n_edges, n_nodes], index=['tissue', 'n_edges', 'n_nodes']).T,
                                pd.DataFrame([[k, table.shape[0],
                                                len(list(set(table['protein1'].values) | set(table['protein2'].values)))] for
                                            k, table in df.groupby('tissue')[['protein1', 'protein2']]],
                                            columns=['tissue', 'n_edges', 'n_nodes'])], axis=0).set_index(
            'tissue').sort_values(by=['n_edges', 'n_nodes'], ascending=False)

        # save data set if possible
        if save_data:
            joblib.dump(df_stats, os.path.join(path, 'df_stats.joblib'))
    return df_stats


### helper function to construct the graph from the data set
def construct_graph(df: pd.DataFrame, tissue: str='total'):
    # construct the network using the data frame containing the data and subset if a specific tissue was selected
    if tissue=='total':
        return nx.from_pandas_edgelist(df, source='protein1', target='protein2')
    else:
        return nx.from_pandas_edgelist(df.loc[df['tissue']==tissue], source='protein1', target='protein2')


### get the LCC from the constructed graph
def get_LCC(G: nx.Graph):
    return G.subgraph(max(nx.connected_components(G), key=len)).copy()


### perform the initial analysis of the graph
def perform_initial_analysis(G: nx.Graph):
    # connected components
    n_connected_comp = nx.number_connected_components(G)
    size_connected_comps = [len(elem) for elem in sorted(nx.connected_components(G), key=len, reverse=True)]

    # largest connected component
    LCC = get_LCC(G)
    n_nodes_in_LCC = size_connected_comps[0]
    percentage_nodes_in_LCC = n_nodes_in_LCC / len(G.nodes)
    n_edges_in_LCC = len(LCC.edges)
    percentage_edges_in_LCC = n_edges_in_LCC / len(G.edges)

    # average distance in LCC
    avg_dist_in_LCC = nx.average_shortest_path_length(LCC)

    # average degree in LCC
    avg_degree_in_LCC = sum(dict(LCC.degree()).values()) / n_nodes_in_LCC

    # average clustering coefficient in LCC
    avg_clustering_coeff_in_LCC = sum(dict(nx.clustering(LCC)).values()) / n_nodes_in_LCC

    # diameter (shortest longest path) in LCC
    diameter_in_LCC = nx.diameter(LCC)

    # degree assortativity coefficient in LCC
    degree_asssortativity_coeff_in_LCC = nx.degree_assortativity_coefficient(LCC)

    # average neighbor degree in LCC
    avg_neighbor_degree_in_LCC = sum(dict(nx.average_neighbor_degree(LCC)).values()) / n_nodes_in_LCC

    # store all results in a pd.Series object and return it
    X_prim_ana_summary = pd.Series(
        [len(G.nodes), len(G.edges), n_connected_comp, n_nodes_in_LCC, percentage_nodes_in_LCC, n_edges_in_LCC,
         percentage_edges_in_LCC, avg_dist_in_LCC, avg_degree_in_LCC, avg_neighbor_degree_in_LCC, avg_clustering_coeff_in_LCC, diameter_in_LCC,
         degree_asssortativity_coeff_in_LCC], index=pd.Index(
            ['number of nodes in G', 'number of edges in G', 'number of connected components', 'number of nodes in LCC',
             'percentage nodes in LCC vs. G', 'number of edges in LCC', 'percentage edges in LCC vs. G',
             'average distance in LCC', 'average degree in LCC', 'average neighbor degree in LCC',
             'average clustering coefficient in LCC', 'diameter of LCC', 'degree assortativity coefficient of LCC']))
    return X_prim_ana_summary
