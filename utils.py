"""
Utility functions for the GNN_prediction project.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

def save_results_to_csv(all_results, filename='training_results.csv'):
    """
    Save training results to a CSV file.

    Args:
        all_results (dict): Dictionary containing results for each y-label.
        filename (str): Name of the output CSV file.

    Returns:
        DataFrame: Combined results DataFrame.
    """
    results_df = pd.concat(all_results.values(), keys=all_results.keys())
    results_df.to_csv(filename, index=True)
    print(f"Results saved to {filename}")
    return results_df

def plot_results(results_df, y_label, output_dir='results_plots'):
    """
    Plot training and validation results.

    Args:
        results_df (DataFrame): DataFrame containing the results to plot.
        y_label (str): The y-label for which the results are plotted.
        output_dir (str): Directory where the plot will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(results_df['Epoch'], results_df['Train_Loss'], label='Train Loss')
    plt.plot(results_df['Epoch'], results_df['Val_Loss'], label='Validation Loss')
    plt.title(f'{y_label} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(results_df['Epoch'], results_df['Train_R²'], label='Train R²')
    plt.plot(results_df['Epoch'], results_df['Val_R²'], label='Validation R²')
    plt.title(f'{y_label} - R² Score')
    plt.xlabel('Epoch')
    plt.ylabel('R² Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{y_label}_plot.png'))
    plt.close()
    print(f"Plot saved for {y_label}")