import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_combined_loss(all_results, output_dir='results_plots'):
    num_labels = len(all_results)
    num_cols = 4  # Number of columns
    num_rows = (num_labels + num_cols - 1) // num_cols  # Calculate required rows
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 5 * num_rows), squeeze=False)
    
    for idx, (label, results_df) in enumerate(all_results.items()):
        row = idx // num_cols
        col = idx % num_cols
        
        axes[row, col].plot(results_df['Epoch'], results_df['Train_Loss'], label='Train Loss', color='blue')
        axes[row, col].plot(results_df['Epoch'], results_df['Val_Loss'], label='Validation Loss', color='orange')
        #axes[row, col].set_title(f'{label} - Loss')
        axes[row, col].set_xlabel('Epoch')
        axes[row, col].set_ylabel(f'{label} Loss')
        axes[row, col].legend()

    # Hide empty subplots if any
    for i in range(num_labels, num_rows * num_cols):
        axes[i // num_cols, i % num_cols].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
    plt.close()
    print(f"Combined loss plot saved for all labels.")

def plot_combined_r2(all_results, output_dir='results_plots'):
    num_labels = len(all_results)
    num_cols = 4  # Number of columns
    num_rows = (num_labels + num_cols - 1) // num_cols  # Calculate required rows
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 5 * num_rows), squeeze=False)
    
    for idx, (label, results_df) in enumerate(all_results.items()):
        row = idx // num_cols
        col = idx % num_cols
        
        axes[row, col].plot(results_df['Epoch'], results_df['Train_R²'], label='Train R²', color='green')
        axes[row, col].plot(results_df['Epoch'], results_df['Val_R²'], label='Validation R²', color='red')
        #axes[row, col].set_title(f'{label} - R² Score')
        axes[row, col].set_xlabel('Epoch')
        axes[row, col].set_ylabel(f'{label} R² Score')
        axes[row, col].legend()

    # Hide empty subplots if any
    for i in range(num_labels, num_rows * num_cols):
        axes[i // num_cols, i % num_cols].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'r2_plot.png'))
    plt.close()
    print(f"Combined R² plot saved for all labels.")


# Function to plot true vs. predicted values for each label
def plot_true_vs_predicted(labels, output_dir='results_plots'):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(20, 3 * len(labels) // 4 + 5))

    for i, label in enumerate(labels):
        # Load the true and predicted values from CSV
        df = pd.read_csv(f'./{output_dir}/predictions_{label}.csv')
        
        # Plotting
        plt.subplot((len(labels) + 3) // 4, 4, i + 1)
        plt.scatter(df['True'], df['Predicted'], alpha=0.5)
        plt.plot([df['True'].min(), df['True'].max()], [df['True'].min(), df['True'].max()], 'r--')  # Line y=x
        #plt.title(f'{label} - True vs Predicted')
        plt.xlabel(f'True {label}')
        plt.ylabel(f'Predicted {label}')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'true_vs_predicted.png'))
    plt.close()
    print(f"True vs. Predicted plots saved to {os.path.join(output_dir, 'true_vs_predicted_all_labels.png')}.")

