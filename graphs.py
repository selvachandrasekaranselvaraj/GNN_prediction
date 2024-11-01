import warnings
warnings.filterwarnings('ignore')

import os
import pickle
import random
import torch
from torch_geometric.data import Data
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats as scipy_stats

def convert_graph_to_data(g, y_value):
    """
    Convert a single graph to a Data object.
    """
    # Convert node features
    x = torch.tensor(g['node_features'], dtype=torch.float)

    # Convert edge indices
    edge_index = torch.tensor(g['edge_indices'], dtype=torch.long).t().contiguous()

    # Convert edge features
    edge_attr = torch.tensor(g['edge_features'], dtype=torch.float).view(-1, 1)

    # Convert y_labels, using only valid indices
    y = torch.tensor([y_value], dtype=torch.float)

    # Create Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    return data

def convert_graphs_to_data_list(graphs, y_values_list):
    data_list = [convert_graph_to_data(g, y_value) for g, y_value in tqdm(zip(graphs, y_values_list), desc="Graphs2data")]
    return data_list

def clean_graphs_for_y_label(values, label):
    values = np.array([v if v is not None else np.nan for v in values])
    # Filter based on label-specific conditions
    if label in ['B1', 'B2', 'B3']:
        selected_indices = np.where((10 < values) & (values < 410) & (~np.isnan(values)))[0]
    elif label in ['G1', 'G2', 'G3']:
        selected_indices = np.where((10 < values) & (values < 230) & (~np.isnan(values)))[0]
    elif label == 'V':
        selected_indices = np.where((0 < values) & (values < 50) & (~np.isnan(values)))[0]
    elif label == 'n':
        selected_indices = np.where((0 < values) & (values < 50) & (~np.isnan(values)))[0]
    elif label == 'EAH':
        selected_indices = np.where((0 < values) & (values < 0.6) & (~np.isnan(values)))[0]
    elif label == 'Eg':
        selected_indices = np.where((0 < values) & (values < 5) & (~np.isnan(values)))[0]
    elif label in ['UE', 'E']:
        selected_indices = np.where((-40 < values) & (values < 0) & (~np.isnan(values)))[0]
    elif label == 'FE':
        selected_indices = np.where((-5 < values) & (values < 3) & (~np.isnan(values)))[0]
    elif label in ['VBM', 'CBM', 'Ef']:
        selected_indices = np.where((-5.1 < values) & (values < 12) & (~np.isnan(values)))[0]
    elif label in ['CN', 'is_S']:
        selected_indices = ~np.isnan(values)  # No specific filtering
    else:
        selected_indices = ~np.isnan(values)  # No specific filtering

    return selected_indices 

def read_graphs(output_file):
    try:
        with open(output_file, 'rb') as f:
            graphs = []
            while True:
                try:
                    graph = pickle.load(f)
                    graphs.append(graph)
                except EOFError:
                    break
        print(f"Number of graphs in {output_file}: {len(graphs)}")
        return graphs
    except FileNotFoundError:
        print(f"File not found: {output_file}")
        return []  # Returning an empty list instead of 0 for consistency in return type
    except TypeError as e:
        print(f"TypeError encountered: {e}")
        return []

def process_graphs():
    crystal_types = ['cubic', 'hexagonal', 'trigonal', 'tetragonal', 'orthorhombic', 'monoclinic', 'triclinic']
    feature_types = ['gnn']
    graphs = []

    for crystal_type in crystal_types:
        for feature_type in feature_types:
            output_file = f'../graphs/graphs_{feature_type}_{crystal_type}.pkl'
            graph_ = read_graphs(output_file)
            graphs.extend(graph_)

    plot_labels = ['V', 'n', 'ρ', 'B1', 'B2', 'B3', 'G1', 'G2', 'G3', 'CN', 'SGN', 'UE', 'E', 'FE', 'EAH', 'is_S', 'Eg', 'CBM', 'VBM', 'Ef']
    print(f"Original number of graphs: {len(graphs)}")
    
    y_values_dict = {label: [] for label in plot_labels}

    for g in graphs:
        for i, label in enumerate(plot_labels):
            value = g['y_values'][i]
            y_values_dict[label].append(value)

    return y_values_dict, graphs

def main():
    y_values_dict, graphs = process_graphs()
   
    for i, (label, values) in enumerate(y_values_dict.items()):
        y_values = np.array(values)  # y_values of label in plot_labels
        selected_indices = clean_graphs_for_y_label(y_values, label)
        clean_graphs = np.array(graphs)[selected_indices]
        clean_y = y_values[selected_indices]

        graphs_data = convert_graphs_to_data_list(clean_graphs, clean_y)

        print(f"Out of {len(graphs)}, number of Data objects created for {label}: {len(graphs_data)}")

if __name__ == '__main__':
    main()

