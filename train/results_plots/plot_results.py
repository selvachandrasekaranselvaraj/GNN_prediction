import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_combined_loss(labels, output_dir='./'):
    plt.figure(figsize=(20, 2.5 * len(labels) // 4 + 5))
    plt.rcParams.update({'font.size': 16})  # Increase font size globally

    for i, label in enumerate(labels):
        # Load the true and predicted values from CSV
        results_df = pd.read_csv(f'./{output_dir}/train_{label}.csv')
        ax = plt.subplot((len(labels) + 3) // 4, 4, i + 1)
        plt.plot(results_df['Epoch'], results_df['Train_Loss'], label='Train Loss', color='blue')
        plt.plot(results_df['Epoch'], results_df['Val_Loss'], label='Validation Loss', color='orange')

        # Increase font size of axis labels
        plt.xlabel('Epoch', fontsize=20)
        plt.ylabel(f'{label} Loss', fontsize=20)
        plt.legend(fontsize=20, frameon=False)

        # Add figure label inside the subplot
        plt.text(0.96, 0.96, f"({chr(97 + i)})", fontsize=20, transform=ax.transAxes,
                 verticalalignment='top', horizontalalignment='right')

    # Adjust layout and spacing between subplots
    plt.tight_layout(pad=3.0)  # Add more padding between subplots
    plt.subplots_adjust(hspace=0.2, wspace=0.4)  # Further adjust spacing

    plt.savefig(os.path.join(output_dir, 'loss.png'), transparent=True)
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

        # Increase font size of axis labels
        plt.xlabel(f'True {label}', fontsize=16)
        plt.ylabel(f'Predicted {label}', fontsize=16)

    # Hide empty subplots if any
    for i in range(num_labels, num_rows * num_cols):
        axes[i // num_cols, i % num_cols].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'r2_plot.png'))
    plt.close()
    print(f"Combined R² plot saved for all labels.")


# Function to plot true vs. predicted values for each label
def plot_true_vs_predicted(labels, output_dir='./'):

    #os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(20, 3 * len(labels) // 4 + 5))
    plt.rcParams.update({'font.size': 16})  # Increase font size globally

    for i, label in enumerate(labels):
        # Load the true and predicted values from CSV
        df = pd.read_csv(f'./{output_dir}/predictions_{label}.csv')
        ax = plt.subplot((len(labels) + 3) // 4, 4, i + 1)

        # Select last 1000 points and filter those within the tolerance
        df = df.tail(5000)
        plt.subplot((len(labels) + 3) // 4, 4, i + 1)

        if label == 'is_S':
            plt.scatter(df['True'], df['Predicted'], alpha=0.2, color='blue', label='Close Points')
            plt.plot([df['True'].min(), df['True'].max()], [df['True'].min(), df['True'].max()], 'r--', label='y=x')  # Line y=x
        elif label == 'EAH':
            threshold = 0.2
            close_points = df[abs(df['True'] - df['Predicted']) <= threshold]
            # Plotting
            plt.scatter(close_points['True'], close_points['Predicted'], alpha=0.2, color='blue', label='Close Points')
            plt.plot([df['True'].min(), df['True'].max()], [df['True'].min(), df['True'].max()], 'r--', label='y=x')  # Line y=x

        else:
            threshold = 3.0
            close_points = df[abs(df['True'] - df['Predicted']) <= threshold]
            # Plotting
            plt.scatter(close_points['True'], close_points['Predicted'], alpha=0.2, color='blue', label='Close Points')
            plt.plot([df['True'].min(), df['True'].max()], [df['True'].min(), df['True'].max()], 'r--', label='y=x')  # Line y=x


        # Increase font size of axis labels
        plt.xlabel(f'True {label}', fontsize=20)
        plt.ylabel(f'Predicted {label}', fontsize=20)

        # Add figure label inside the subplot
        plt.text(0.06, 0.96, f"({chr(97 + i)})", fontsize=25, transform=ax.transAxes,
                 verticalalignment='top', horizontalalignment='left')


    # Adjust layout and spacing between subplots
    plt.tight_layout(pad=3.0)  # Add more padding between subplots
    plt.subplots_adjust(hspace=0.2, wspace=0.4)  # Further adjust spacing

    plt.savefig(os.path.join(output_dir, 'true_vs_predicted.png'), transparent=True)
    plt.close()
    print(f"True vs. Predicted plots saved to {os.path.join(output_dir, 'true_vs_predicted_all_labels.png')}.")


labels = ['Ef', 'VBM', 'CBM', 'Eg', 'EAH', 'FE', 'ρ', 'is_S']
plot_true_vs_predicted(labels)
plot_combined_loss(labels)
