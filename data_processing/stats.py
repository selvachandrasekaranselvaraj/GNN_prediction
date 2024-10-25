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

def convert_graph_to_data(g, valid_y_indices):
    y_labels = [g['y_labels'][i] for i in valid_y_indices]
    y = torch.tensor(y_labels, dtype=torch.float)
    return Data(x=None, edge_index=None, edge_attr=None, y=y)

def analyze_y_label_statistics(data_list, y_idx, y_label):
    y_values = [data.y[y_idx].item() for data in data_list]
    
    # Calculate statistics
    mean = np.mean(y_values)
    median = np.median(y_values)
    std_dev = np.std(y_values)
    variance = np.var(y_values)
    skewness = scipy_stats.skew(y_values)
    kurtosis = scipy_stats.kurtosis(y_values)
    _, normality_p_value = scipy_stats.normaltest(y_values)

    # Find the 5th and 95th percentiles
    lower_bound = np.percentile(y_values, 5)
    upper_bound = np.percentile(y_values, 95)

    # Filter out the minority data
    filtered_y_values = [y for y in y_values if lower_bound <= y <= upper_bound]

    # Plotting histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(y_values, kde=True, bins=100, stat='percent')
    plt.title(f"Distribution of {y_label}")
    plt.xlabel(y_label)
    plt.ylabel("Percentage")
    plt.savefig(f"{y_label}_distribution_filtered.png")
    plt.close()

    # Plotting histogram with full data for comparison
    plt.figure(figsize=(10, 6))
    sns.histplot(y_values, kde=True, bins=100, stat='percent')
    plt.title(f"Distribution of {y_label}")
    plt.xlabel(y_label)
    plt.ylabel("Percentage")
    plt.savefig(f"{y_label}_distribution_full.png")
    plt.close()

    return {
        'Mean': mean,
        'Median': median,
        'Standard Deviation': std_dev,
        'Variance': variance,
        'Skewness': skewness,
        'Kurtosis': kurtosis,
        'Normality p-value': normality_p_value
    }

def analyze_covariance(y_values_dict):
    labels = list(y_values_dict.keys())
    data = np.array([y_values_dict[label] for label in labels]).T

    cov_matrix = np.cov(data.T)
    corr_matrix = np.corrcoef(data.T)

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', xticklabels=labels, yticklabels=labels)
    plt.title("Correlation Matrix of Y Labels")
    plt.tight_layout()
    plt.savefig("y_labels_correlation_matrix.png")
    plt.close()

    plt.figure(figsize=(12, 10))
    sns.heatmap(cov_matrix, annot=True, cmap='viridis', xticklabels=labels, yticklabels=labels)
    plt.title("Covariance Matrix of Y Labels")
    plt.tight_layout()
    plt.savefig("y_labels_covariance_matrix.png")
    plt.close()


def plot_statistics(stats_dict):
    stat_names = ['Mean', 'Median', 'Standard Deviation', 'Variance', 'Skewness', 'Kurtosis']
    y_labels = list(stats_dict.keys())
    
    # Create a DataFrame for easier plotting
    data = []
    for y_label in y_labels:
        for stat in stat_names:
            data.append({
                'Y Label': y_label,
                'Statistic': stat,
                'Value': stats_dict[y_label][stat]
            })
    df = pd.DataFrame(data)
    
    # Create the plot
    plt.figure(figsize=(15, 10))
    sns.barplot(x='Statistic', y='Y Label', hue='Statistic', data=df, dodge=False)
    
    plt.title('Statistical Measures for Y Labels', fontsize=16)
    plt.xlabel('Statistic', fontsize=12)
    plt.ylabel('Y Label', fontsize=12)
    plt.legend(title='Statistic', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('y_label_statistics_horizontal.png', bbox_inches='tight')
    plt.close()

    # Create individual plots for each statistic
    for stat in stat_names:
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Value', y='Y Label', data=df[df['Statistic'] == stat], orient='h')
        
        plt.title(f'{stat} for Y Labels', fontsize=16)
        plt.xlabel('Value', fontsize=12)
        plt.ylabel('Y Label', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f'y_label_{stat.lower().replace(" ", "_")}.png')
        plt.close()

def analyze_stats(graphs):
    y_label_names = ['volume',
            'nsites',
            'density',
            'bulk_modulus',
            'shear_modulus',
            'crystal_number',
            'space_group_number',
            'uncorrected_energy_per_atom',
            'energy_per_atom',
            'formation_energy_per_atom',
            'energy_above_hull',
            'is_stable',
            'band_gap',
            'cbm',
            'vbm',
            'efermi']

    valid_y_indices = [] #np.arange(len(y_label_names))#[]
    for i, label_name in enumerate(y_label_names):
        if all(g['y_labels'][i] is not None for g in graphs):
            valid_y_indices.append(i)

    valid_y_labels = [y_label_names[i] for i in valid_y_indices]
    print(f"Valid y_labels: {valid_y_labels}")

    data_list = [convert_graph_to_data(g, valid_y_indices) for g in tqdm(graphs, desc="Converting graphs to data objects")]

    y_values_dict = {}
    stats_dict = {}
    
    for y_idx, y_label in enumerate(valid_y_labels):
        y_stats = analyze_y_label_statistics(data_list, y_idx, y_label)
        stats_dict[y_label] = y_stats
        y_values_dict[y_label] = [data.y[y_idx].item() for data in data_list]
    
    analyze_covariance(y_values_dict)
    plot_statistics(stats_dict)  # New function to plot statistics as subplots

    return



def process_graphs():
    crystal_types = ['cubic', 'hexagonal', 'trigonal', 'tetragonal', 'orthorhombic', 'monoclinic', 'triclinic']
    feature_type = 'gnn'
    graphs = []

    for crystal_type in tqdm(crystal_types, desc="Reading graph files"):
        graphs.extend(read_graph_file(crystal_type, feature_type))

    print(f"Total number of graphs loaded: {len(graphs)}")
    random.shuffle(graphs)
    return graphs

def read_graph_file(crystal_type, feature_type):
    output_file = f'../graphs/graphs_{feature_type}_{crystal_type}.pkl'
    try:
        with open(output_file, 'rb') as f:
            graphs = []
            while True:
                try:
                    graphs.append(pickle.load(f))
                except EOFError:
                    break
        return graphs
    except FileNotFoundError:
        print(f"File not found: {output_file}")
        return []

def main():
    graphs = process_graphs()
    analyze_stats(graphs)

if __name__ == '__main__':
    main()
