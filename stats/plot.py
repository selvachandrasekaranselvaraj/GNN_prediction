"""
This module provides functions for plotting various statistical visualizations for the GNN_prediction project.

It includes functions to plot distribution plots, correlation matrices, and covariance matrices of the dataset.

Functions:
    clean_data: Helper function to clean the input data and return a DataFrame.
    plot_matrices: Plot both covariance and correlation matrices.
    plot_correlation_matrix: Plot the correlation matrix of the dataset.
    plot_covariance_matrix: Plot the covariance matrix of the dataset.
    plot_distributions: Plot distribution plots for each property in the dataset.
    main: The main function to generate all plots.

This module uses libraries such as Matplotlib, Seaborn, and Pandas for data visualization.
Note: This module suppresses warnings to avoid cluttering the output.
"""

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
import seaborn as sns
from scipy import stats as scipy_stats
from matplotlib.ticker import ScalarFormatter


def clean_data(y_values_dict):
    """ Helper function to clean the input data and return a DataFrame. """
    cleaned_dict = {k: v for k, v in y_values_dict.items() if any(x is not None for x in v)}
    return pd.DataFrame(cleaned_dict).dropna()

def plot_matrices(y_values_dict):
    df = clean_data(y_values_dict)

    # Calculate covariance and correlation matrices
    cov_matrix = df.cov()
    corr_matrix = df.corr()

    # Create a two-column subplot
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    plt.rcParams.update({'font.size': 20})

    # Plot covariance matrix
    sns.heatmap(cov_matrix, annot=False, cmap='viridis',
                xticklabels=cov_matrix.columns, yticklabels=cov_matrix.columns,
                vmin=-1e2, vmax=1e2, ax=axes[0])
    axes[0].set_title("(a)", fontsize=30)
    
    # Set font size for x and y tick labels and avoid overlap
    axes[0].set_xticklabels(axes[0].get_xticklabels(), fontsize=20, rotation=90, ha='right')  # Rotate x-tick labels
    axes[0].set_yticklabels(axes[0].get_yticklabels(), fontsize=20, rotation=0)  # Keep y-tick labels horizontal


    # Plot correlation matrix
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm',
                xticklabels=corr_matrix.columns, yticklabels=corr_matrix.columns, ax=axes[1])
    axes[1].set_title("(b)", fontsize=30)
    
    # Set font size for x and y tick labels and avoid overlap
    axes[1].set_xticklabels(axes[1].get_xticklabels(), fontsize=20, rotation=90, ha='right')  # Rotate x-tick labels
    axes[1].set_yticklabels(axes[1].get_yticklabels(), fontsize=20, rotation=0)  # Keep y-tick labels horizontal


    plt.tight_layout()
    plt.savefig("covariance_correlation_matrices.png", transparent=True)
    plt.close()

def plot_correlation_matrix(y_values_dict):
    df = clean_data(y_values_dict)
    
    # Calculate correlation matrix
    corr_matrix = df.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', 
                xticklabels=corr_matrix.columns, yticklabels=corr_matrix.columns)
    plt.title("(a) Correlation Matrix", fontsize=16)
    plt.tight_layout()
    plt.savefig("correlation_matrix.png")
    plt.close()

def plot_covariance_matrix(y_values_dict):
    df = clean_data(y_values_dict)
    
    # Calculate covariance matrix
    cov_matrix = df.cov()

    plt.figure(figsize=(12, 10))
    sns.heatmap(cov_matrix, annot=False, cmap='viridis', 
                xticklabels=cov_matrix.columns, yticklabels=cov_matrix.columns,
                vmin=-1e2, vmax=1e2)
    plt.title("(b) Covariance Matrix", fontsize=16)
    plt.tight_layout()
    plt.savefig("covariance_matrix.png")
    plt.close()

def plot_distributions(y_values_dict):
    df = clean_data(y_values_dict)
    
    num_labels = len(df.columns)  # Get the number of labels (columns)
    num_cols = 4
    num_rows = (num_labels + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 12))
    nbins = 300
    for idx, (label, values) in enumerate(df.items()):
        row = idx // num_cols
        col = idx % num_cols
        ax = axes[row, col] if num_rows > 1 else axes[col]

        # Remove None values
        values = np.array([v for v in values if v is not None])
        
        # Adjust bin number based on value ranges and specific labels
        if label in ['B1', 'B2', 'B3']:
            x = values[(10 < values) & (values < 410)]
        elif label in ['G1', 'G2', 'G3']:
            x = values[(10 < values) & (values < 230)]
        elif label == 'V':
            x = values[(0 < values) & (values < 50)]
        elif label == 'n':
            x = values[(0 < values) & (values < 50)]
        elif label == 'EAH':
            x = values[(0 < values) & (values < 0.6)]
        elif label == 'Eg':
            x = values[(0 < values) & (values < 5)]
        elif label in ['UE', 'E']:
            x = values[(-40 < values) & (values < 0)]
        elif label == 'FE':
            x = values[(-5 < values) & (values < 3)]
        elif label in ['VBM', 'CBM', 'Ef']:
            x = values[(-5.1 < values) & (values < 12)]
        elif label in ['CN', 'is_S']:
            x = values
        else:
            x = values

        # Determine number of bins for histogram
        if len(x) < 300:
            nbins = int(len(x) + (len(x) * 0.1))
        else:
            nbins = 300 
        sns.histplot(x, kde=True, bins=nbins, stat='density', ax=ax)
        ax.set_xlabel(label, fontsize=12)
        ax.set_ylabel("a.u.", fontsize=12)
        #ax.set_title(f"({chr(97 + idx)})", fontsize=16, loc='left')  # Title for distributions (a), (b), (c), ...

        # Add figure label inside the subplot
        ax.text(0.96, 0.96, f"({chr(97 + idx)})", fontsize=14, transform=ax.transAxes, 
                verticalalignment='top', horizontalalignment='right')

        # Set x and y axis to exponential format
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    # Remove empty subplots
    for idx in range(num_labels, num_rows * num_cols):
        row = idx // num_cols
        col = idx % num_cols
        fig.delaxes(axes[row, col])

    plt.tight_layout()
    plt.savefig("distributions.png", transparent=True)
    plt.close()

def main():
    # Load data from CSV file into a dictionary
    y_values_dict = pd.read_csv('y_values_dict.csv')
    
    # Ensure y_values_dict is a dictionary if not already
    y_values_dict = y_values_dict.to_dict(orient='list') 

    print(len(y_values_dict))  # Check the number of items
    plot_distributions(y_values_dict)
    #plot_correlation_matrix(y_values_dict)
    #plot_covariance_matrix(y_values_dict)
    plot_matrices(y_values_dict)

if __name__ == '__main__':
    main()