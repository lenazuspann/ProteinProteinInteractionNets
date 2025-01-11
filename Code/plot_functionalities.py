import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Optional, List
import networkx as nx
from scipy.stats import linregress

from analysis_functionalities import construct_graph, get_LCC
from settings import *


### visualize the graph using plotly
def get_graph_visualization(G: nx.Graph, save_fig: bool=True, extra_label: str='', path: str=path_plots):
    # construct a layout based on the graph such that the plot fits in the frame 
    pos = nx.spring_layout(G)

    # save the edge data for the plot
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    # construct the plot of the edges
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    # save the node data for the plot
    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    # construct the plot of the nodes
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    # interactive part: get the number of connections of the nodes by hovering over them
    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append('# of connections: ' + str(len(adjacencies[1])))

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    # construct the graphic
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Graph visualization for '+extra_label,
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    
    if save_fig:
        fig.write_image(os.path.join(path, 'graph_' + extra_label + '.png'))
    else:
        fig.show()


### plot the degree distribution histogram
def get_degree_distribution_plot(df: pd.DataFrame, df_init_ana: pd.DataFrame, tissue_list: List, save_fig: bool=True, log: bool=True, scale: Optional[float]=None, path: str=path_plots):
    # initialze the plotly figure to be displayed
    fig = go.Figure()
    
    # calculate the histogram data from the networks degrees
    hist_data = [list(np.unique(sorted((d for n, d in construct_graph(df=df, tissue=tissue).degree()), reverse=True),
                                return_counts=True)) for tissue in tissue_list]
    
    # truncate the data if the scale parameter is set to a usable value
    if scale is not None:
        hist_data = [[[k for idx, k in enumerate(hist_data[i][0]) if
                           hist_data[i][1][idx] >= scale * df_init_ana.loc['average degree in LCC'][tissue_list[i]]],
                          [j for j in hist_data[i][1] if
                           j >= scale * df_init_ana.loc['average degree in LCC'][tissue_list[i]]]] for i in
                         range(len(hist_data))]
    
    # add the line for the respective degree distribution plots per tissue
    for i, tissue in enumerate(tissue_list):
        fig.add_trace(
            go.Scatter(x=hist_data[i][0], y=hist_data[i][1], mode='lines+markers',
                       name=tissue, marker=dict(colorscale='YlGnBu')))
    
    # set some style parameters
    fig.update_layout(xaxis_title='degree', yaxis_title='count',
                      title='Degree distribution plot (scale=' +  str(scale) + ', log=' + str(log) + ')')
    
    # change to logarithmic axes if the parameter is set
    if log:
        fig.update_yaxes(type="log")
        fig.update_xaxes(type="log")
    
    if save_fig:
        fig.write_image(os.path.join(path,'degree_distribution_scale=' + str(scale) + '_log=' + str(log) + '.png'))
    else:
        fig.show()


### histogram with logarithmic binning and fitted line
def get_log_hist_with_fitted_line(G: nx.Graph, save_fig: bool=True, path: str=path_plots):
    # get the degrees from the network
    degrees = [degree for node, degree in G.degree()]
    
    # calclate the data for the histogram
    degree_counts, bin_edges = np.histogram(degrees, bins=range(1, max(degrees) + 2), density=True)
    
    # construct the logarithmic bins
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    np.seterr(divide='ignore') # note: we deal with this problem below, that is why we can ignore it here
    log_bin_centers = np.log10(bin_centers)
    log_degree_counts = np.log10(degree_counts)
    
    # note: if the data has values close to zero, we need to exclude those values to proceed
    valid = np.isfinite(log_bin_centers) & np.isfinite(log_degree_counts)
    
    # fit the line
    slope, intercept, r_value, p_value, std_err = linregress(log_bin_centers[valid], log_degree_counts[valid])
    alpha = -slope
    print(f"The power law exponent is {alpha:.2f}")
    
    # initialize the plot
    fig = go.Figure()
    
    # add scatter plot for the log-log degree distribution
    fig.add_trace(go.Scatter(
        x=log_bin_centers[valid],
        y=log_degree_counts[valid],
        mode='markers',
        name='Data'
    ))
    
    # add the fitted line
    fig.add_trace(go.Scatter(
        x=log_bin_centers[valid],
        y=intercept + slope * log_bin_centers[valid],
        mode='lines',
        name=f'Fit: slope = {slope:.2f}'
    ))
    
    # customize the layout
    fig.update_layout(
        title='Log-Log Degree Distribution and Power Law Fit',
        xaxis_title='log(Degree)',
        yaxis_title='log(Frequency)',
        showlegend=True
    )

    if save_fig:
        fig.write_image(os.path.join(path,'hist_log_binning_fitted_line.png'))
    else:
        fig.show()


### plot average local clustering coefficient against the node degree
def get_avg_clustering_coeff_vs_degree_plot(G: nx.Graph, save_fig: bool=True, path: str=path_plots):
    # get the ordered degree sequence from the input graph
    degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
    k_max = max(degree_sequence)
    
    # initialize lists to store values to plot
    list_degrees = []
    list_av_clustering_coeffs = []
    
    # caluculate the average local clustering coefficient of nodes with the same degree
    for k in range(k_max):
        nodes_degree_k = [v for v in G.nodes() if G.degree(v) == k]
        clustering_coeff = nx.clustering(G, nodes_degree_k)
        if len(clustering_coeff) != 0:
            list_av_clustering_coeffs.append(sum(clustering_coeff.values()) / len(clustering_coeff))
            list_degrees.append(k)
    
    # plot the values in a scatter plot
    fig = go.Figure(data=go.Scatter(x=list_degrees, y=list_av_clustering_coeffs, mode='lines+markers', marker=dict(colorscale='YlGnBu')))
    fig.update_layout(xaxis_title='node degree', yaxis_title='average clustering coefficient',
                      title='Average clustering coefficient against node degree')
    
    if save_fig:
        fig.write_image(os.path.join(path, 'avg_clustering_vs_degree.png'))
    else:
        fig.show()


### distance distribution plot
def get_distance_distribution_plot(G: nx.Graph, cutoff: int, save_fig: bool=True, path: str=path_plots):
    # calculate distance values and count how many pairs of nodes has this length as shortest path between them
    dict_lengths = dict(nx.all_pairs_shortest_path_length(G, cutoff=cutoff))
    list_lengths = [elem for d in dict_lengths.values() if isinstance(d, dict) for elem in d.values() if elem >0]
    
    # calculate mean and variance and print them
    mean_dist = np.average(list_lengths)
    var_dist = np.var(list_lengths)
    print(f'The average shortest path length in the network is {mean_dist:.2f}.')
    print(f'The variance from the average shortest path length in the network is {var_dist:.2f}.')
    
    # sequence of lengths nedded for the plot
    length_seq = list(np.unique(sorted(list_lengths, reverse=True), return_counts=True))
    
    # construct the scatter plot and customize the layout
    fig = go.Figure(data=go.Scatter(x=length_seq[0], y=[elem/sum(length_seq[1]) for elem in length_seq[1]], mode='lines+markers', marker=dict(colorscale='YlGnBu')))
    fig.update_layout(xaxis_title='distance (length of shortest path)', yaxis_title='count',
                      title='Distance distribution function')
    
    if save_fig:
        fig.write_image(os.path.join(path, 'distance_distribution.png'))
    else:
        fig.show()


### plot the degree correlation funciton
def get_degree_correlation_fct(G: nx.Graph, save_fig: bool=True, path: str=path_plots):
    average_neighbor_degrees = nx.average_neighbor_degree(G)
    
    # extract node degrees and average neighbor degrees corresponding to that degree
    node_degrees = dict(G.degree())
    x = list(node_degrees.values())
    y = [average_neighbor_degrees[node] for node in G.nodes()]
    
    # create a scatter plot and use x also as the color for better visualization
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=dict(size=10, color=x, colorscale='Viridis', showscale=True), text=list(G.nodes), hoverinfo='text'))
    
    # add diagonal line for reference
    fig.add_trace(go.Scatter(x=[min(x), max(x)], y=[min(x), max(x)], mode='lines', line=dict(color='red', dash='dash'), showlegend=False))
    
    # customize layout
    fig.update_layout(title='Degree correlation function of the network', xaxis_title='node degree', yaxis_title='average neighbor degree',
        coloraxis_colorbar=dict(title='node degree'), hovermode='closest')
    
    if save_fig:
        fig.write_image(os.path.join(path,'degree_correlation_fct.png'))
    else:
        fig.show()










