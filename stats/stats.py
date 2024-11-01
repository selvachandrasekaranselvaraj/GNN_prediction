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
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
import seaborn as sns
from matplotlib.ticker import ScalarFormatter

"""
This module handles statistical analysis and visualization of the dataset for the GNN_prediction project.

It includes functions to process graphs, analyze statistics, and generate various plots to visualize the data distribution and relationships.

The module uses libraries such as NumPy, Pandas, Matplotlib, and Seaborn for data processing and visualization.

Functions:
    main: The main function to run the statistical analysis and generate plots.
    process_graphs: Process and return graphs for analysis.
    analyze_stats: Analyze statistics of the processed graphs and generate plots.

Note: This module suppresses warnings to avoid cluttering the output.
"""

def read_and_count_graphs(output_file):
    try:
        with open(output_file, 'rb') as f:
            graphs = []
            while True:
                try:
                    graph = pickle.load(f)
                    graphs.append(graph)
                except:
                    #except EOFError:
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
            graph_ = None
            #output_file = f'./graphs_{feature_type}_{crystal_type}.pkl'
            output_file = f'../graphs/graphs_{feature_type}_{crystal_type}.pkl'
            graph_ = read_and_count_graphs(output_file)
            graphs.extend(graph_)

    return graphs

def analyze_stats(graphs):
    plot_labels = ['V', 'n', 'ρ', 'B1', 'B2', 'B3', 'G1', 'G2', 'G3', 'CN', 'SGN', 'UE', 'E', 'FE', 'EAH', 'is_S', 'Eg', 'CBM', 'VBM', 'Ef']

    print(f"Original number of graphs: {len(graphs)}")
    y_values_dict = {label: [] for label in plot_labels}

    for g in graphs:
        for i, label in enumerate(plot_labels):
            value = g['y_values'][i]
            y_values_dict[label].append(value)

    print(f"Valid y_labels: {list(y_values_dict.keys())}")

    # Save y_values_dict to a CSV file
    y_values_df = pd.DataFrame(y_values_dict)
    y_values_df.to_csv('y_values_dict.csv', index=False)
    print("y_values_dict saved to y_values_dict.csv")

    return y_values_dict

def main():
    graphs = process_graphs()
    y_values_dict = analyze_stats(graphs)

    # You can add more analysis here if needed

if __name__ == '__main__':
    main()