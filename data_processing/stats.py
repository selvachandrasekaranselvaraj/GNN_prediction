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
            output_file = f'./graphs_{feature_type}_{crystal_type}.pkl'
            #output_file = f'../graphs/graphs_{feature_type}_{crystal_type}.pkl'
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

    plot_distributions(y_values_dict)
    plot_correlation_matrix(y_values_dict)
    plot_covariance_matrix(y_values_dict)

    return y_values_dict

def plot_correlation_matrix(y_values_dict):
    # Remove None values and convert to float
    cleaned_dict = {k: v for k, v in y_values_dict.items() if any(x is not None and x < 2000 for x in v)}
    
    # Create a DataFrame, dropping any remaining NaN values
    df = pd.DataFrame(cleaned_dict).dropna()
    
    # Calculate correlation matrix
    corr_matrix = df.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', xticklabels=corr_matrix.columns, yticklabels=corr_matrix.columns)
    #plt.title("Correlation Matrix of Y Labels")
    plt.tight_layout()
    plt.savefig("correlation_matrix.png")
    plt.close()

def plot_covariance_matrix(y_values_dict):
    # Remove None values and convert to float
    #cleaned_dict = {k: v for k, v in y_values_dict.items() if any(x is not None and x < 2000 for x in v)}
    cleaned_dict = {k: v for k, v in y_values_dict.items() if any(x is not None for x in v)}
    
    # Create a DataFrame, dropping any remaining NaN values
    df = pd.DataFrame(cleaned_dict).dropna()
    
    # Calculate covariance matrix
    cov_matrix = df.cov()

    plt.figure(figsize=(12, 10))

    sns.heatmap(cov_matrix, annot=False, cmap='viridis', 
                xticklabels=cov_matrix.columns, yticklabels=cov_matrix.columns,
                vmin = -1e2, vmax=1e2)  # Set the maximum limit for the color bar
    #plt.title("Covariance Matrix of Y Labels")
    plt.tight_layout()
    plt.savefig("covariance_matrix.png")
    plt.close()

def plot_distributions(y_values_dict):
    #y_values_dict = {k: v for k, v in y_values_dict.items() if any(x is not None and x < 2000 for x in v)}
    num_labels = len(y_values_dict)
    num_cols = 4
    num_rows = (num_labels + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(8, 8)) #figsize=(20, 5 * num_rows))
    #fig.suptitle("Distributions of Y Labels", fontsize=20)
    nbins = 300
    for idx, (label, values) in enumerate(y_values_dict.items()):
        row = idx // num_cols
        col = idx % num_cols
        if label not in ['n', 'CN', 'SGN', 'is_S']:
            ax = axes[row, col] if num_rows > 1 else axes[col]
        
        # Remove None values
        values = np.array([v for v in values if v is not None])
        
        if label in ['B1', 'B2', 'B3']:
            x = values[(10<values)&(values<410)]
            if len(x) < 300:
                nbins = int(len(x) + (len(x)*0.1))
            else:
                nbins = 300 
            sns.histplot(x, kde=True, bins=nbins, stat='density', ax=ax, color='brown')
            print(f"Number of graphs for the data {label}: {len(x)}")

        elif label in ['G1', 'G2', 'G3']:
            x = values[(10<values)&(values<230)]
            if len(x) < 300:
                nbins = int(len(x) + (len(x)*0.1))
            else:
                nbins = 300 
            sns.histplot(x, kde=True, bins=nbins, stat='density', ax=ax, color='brown')
            print(f"Number of graphs for the data {label}: {len(x)}")

        elif label == 'V':
            x = values[(0<values)&(values<50)]
            if len(x) < 300:
                nbins = int(len(x) + (len(x)*0.1))
            else:
                nbins = 300 
            sns.histplot(x, kde=True, bins=nbins, stat='density', ax=ax, color='brown')
            print(f"Number of graphs for the data {label}: {len(x)}")

        elif label == 'n':
            x = values[(0<values)&(values<50)]
            if len(x) < 300:
                nbins = int(len(x) + (len(x)*0.1))
            else:
                nbins = 300 
            #sns.histplot(x, kde=True, bins=nbins, stat='density', ax=ax, color='brown')
            print(f"Number of graphs for the data {label}: {len(x)}")

        elif label== 'EAH':
            x = values[(0<values)&(values<0.6)]
            if len(x) < 300:
                nbins = int(len(x) + (len(x)*0.1))
            else:
                nbins = 300 
            sns.histplot(x, kde=True, bins=nbins, stat='density', ax=ax, color='brown')
            print(f"Number of graphs for the data {label}: {len(x)}")

        elif label== 'Eg':
            x = values[(0<values)&(values<5)]
            if len(x) < 300:
                nbins = int(len(x) + (len(x)*0.1))
            else:
                nbins = 300 
            sns.histplot(x, kde=True, bins=nbins, stat='density', ax=ax, color='brown')
            print(f"Number of graphs for the data {label}: {len(x)}")

        elif label in ['UE', 'E']:
            x = values[(-40<values)&(values<0)]
            if len(x) < 300:
                nbins = int(len(x) + (len(x)*0.1))
            else:
                nbins = 300 
            sns.histplot(x, kde=True, bins=nbins, stat='density', ax=ax, color='brown')
            print(f"Number of graphs for the data {label}: {len(x)}")

        elif label == 'FE':
            x = values[(-5<values)&(values<3)]
            if len(x) < 300:
                nbins = int(len(x) + (len(x)*0.1))
            else:
                nbins = 300 
            sns.histplot(x, kde=True, bins=nbins, stat='density', ax=ax, color='brown')
            print(f"Number of graphs for the data {label}: {len(x)}")

        elif label in ['VBM', 'CBM', 'Ef']:
            x = values[(-5.1<values)&(values<12)]
            if len(x) < 300:
                nbins = int(len(x) + (len(x)*0.1))
            else:
                nbins = 300 
            sns.histplot(x, kde=True, bins=nbins, stat='density', ax=ax, color='brown')
            print(f"Number of graphs for the data {label}: {len(x)}")

        elif label in ['CN', 'is_S', 'SGN']:
            x = values 
            if len(x) < 300:
                nbins = int(len(x) + (len(x)*0.1))
            else:
                nbins = 300 
            #sns.histplot(x, kde=True, bins=7, stat='density', ax=ax, color='brown')
            print(f"Number of graphs for the data {label}: {len(x)}")

        else:
            x = values 
            if len(x) < 300:
                nbins = int(len(x) + (len(x)*0.1))
            else:
                nbins = 300 
            sns.histplot(x, kde=True, bins=nbins, stat='density', ax=ax, color='brown')
            print(f"Number of graphs for the data {label}: {len(x)}")

     
        #ax.set_title(label)
        ax.set_xlabel(label)
        ax.set_ylabel("a.u.")

        # Set x and y axis to exponential format
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        #ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

    # Remove empty subplots
    for idx in range(num_labels, num_rows * num_cols):
        row = idx // num_cols
        col = idx % num_cols
        if num_rows > 1:
            fig.delaxes(axes[row, col])
        else:
            fig.delaxes(axes[col])

    plt.tight_layout()
    plt.savefig("distributions.png")
    plt.close()

def main():
    graphs = process_graphs()
    y_values_dict = analyze_stats(graphs)

    # You can add more analysis here if needed

if __name__ == '__main__':
    main()
