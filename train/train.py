"""
This module contains the main training logic for the GNN_prediction project.

It includes the GNN model definition, training and validation functions, and the cross-validation process.
The module uses PyTorch and PyTorch Geometric for implementing the Graph Neural Network.

Classes:
    GNN: The Graph Neural Network model.

Functions:
    train: Train the model for one epoch.
    validate: Validate the model on the validation set.
    save_results_to_csv: Save training results to a CSV file.
    cross_validate: Perform cross-validation for a given property.
    main: The main function to run the training process.

The module also sets up logging to both a file and console for tracking the training progress.
"""

import warnings
warnings.filterwarnings('ignore')

import os
import pickle
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

from graphs import process_graphs
from graphs import clean_graphs_for_y_label
from graphs import convert_graphs_to_data_list
from plot import plot_combined_r2
from plot import plot_combined_loss, plot_true_vs_predicted

# Configure logging
logging.basicConfig(filename='training_log.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super(GNN, self).__init__()
        self.convs = nn.ModuleList([GCNConv(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)])
        self.lin = nn.Linear(hidden_dim, 1)  # Output a single value
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        mask = (x.sum(dim=-1) != 0)
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x[mask], batch[mask])
        x = self.lin(x)
        return x.squeeze()

def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    all_predictions = []
    all_true_values = []
    
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        all_predictions.append(out.detach().cpu())
        all_true_values.append(data.y.cpu())
    
    all_predictions = torch.cat(all_predictions, dim=0)
    all_true_values = torch.cat(all_true_values, dim=0)
    
    r2 = r2_score(all_true_values, all_predictions)
    return total_loss / len(train_loader), r2

def validate(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_true_values = []
    
    with torch.no_grad():
        for data in val_loader:
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            total_loss += loss.item()
            all_predictions.append(out.cpu())
            all_true_values.append(data.y.cpu())
    
    all_predictions = torch.cat(all_predictions, dim=0)
    all_true_values = torch.cat(all_true_values, dim=0)
    
    r2 = r2_score(all_true_values, all_predictions)
    return total_loss / len(val_loader), r2, all_true_values, all_predictions 

def save_results_to_csv(all_results, filename='training_results.csv'):
    results_df = pd.concat(all_results)
    results_df.to_csv(filename, index=True)
    logging.info(f"Results saved to {filename}")
    return results_df

def cross_validate(data_list, y_label, epochs=1000, batch_size=256, learning_rate=1e-4, patience=10,
                   hidden_dim=200, num_layers=3, dropout=5e-5):

    logging.info(f"\nTraining model for {y_label}")

    y_values = [data.y.item() for data in data_list]
    lower_bound = np.percentile(y_values, 5)
    upper_bound = np.percentile(y_values, 95)
    filtered_data_list = [data for data in data_list if lower_bound <= data.y <= upper_bound]
    logging.info(f"Original data size: {len(data_list)}, Filtered data size: {len(filtered_data_list)}")
    
    train_data, val_data = train_test_split(filtered_data_list, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    input_dim = data_list[0].x.shape[1]
    model = GNN(input_dim, hidden_dim, num_layers, dropout)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    best_model = None
    epochs_no_improve = 0

    train_losses, val_losses = [], []
    train_r2_scores, val_r2_scores = [], []

    # Variables to store true and predicted values for each epoch
    true_values, predicted_values = [], []

    for epoch in range(epochs):
        train_loss, train_r2 = train(model, train_loader, optimizer, criterion)
        val_loss, val_r2, epoch_true_vals, epoch_predicted_vals = validate(model, val_loader, criterion)
        
        # Store true and predicted values for the last epoch
        true_values.extend(epoch_true_vals.cpu().numpy())
        predicted_values.extend(epoch_predicted_vals.cpu().numpy())

        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_r2_scores.append(train_r2)
        val_r2_scores.append(val_r2)
        
        log_msg = (f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                    f'Train R²: {train_r2:.4f}, Val R²: {val_r2:.4f}')
        logging.info(log_msg)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve == patience:
            logging.info(f"Early stopping triggered.")
            break

    models_dir = 'trained_models'
    os.makedirs(models_dir, exist_ok=True)
    torch.save(best_model, f'./{models_dir}/{y_label}.pth')
    logging.info(f"Best model for {y_label} saved.")

    output_dir = 'results_plots'
    os.makedirs(output_dir, exist_ok=True)
    results_df = pd.DataFrame({
        'Epoch': range(1, len(train_losses) + 1),
        'Train_Loss': train_losses,
        'Val_Loss': val_losses,
        'Train_R²': train_r2_scores,
        'Val_R²': val_r2_scores
    })

    # Save true and predicted values to CSV
    predictions_df = pd.DataFrame({
        'True': true_values,
        'Predicted': predicted_values
    })
    predictions_df.to_csv(f'./{output_dir}/predictions_{y_label}.csv', index=False)
    results_df.to_csv(f'./{output_dir}/train_{y_label}.csv', index=False)
    logging.info(f"True and predicted values saved for {y_label}")

    return results_df

def main():
    y_values_dict, graphs = process_graphs()
    all_results = {}  # Dictionary to store results for all labels
    labels = ['Ef', 'VBM', 'CBM', 'Eg', 'EAH', 'FE', 'ρ', 'is_S']
    for i, (label, values) in enumerate(y_values_dict.items()):
        if label in ['Ef', 'VBM', 'CBM', 'Eg', 'EAH', 'FE', 'ρ', 'is_S']:
            y_values = np.array(values)
            selected_indices = clean_graphs_for_y_label(y_values, label)
            clean_graphs = np.array(graphs)[selected_indices]
            clean_y = y_values[selected_indices]
            
            graphs_data = convert_graphs_to_data_list(clean_graphs, clean_y)
            random.shuffle(graphs_data)
            
            logging.info(f"Out of {len(graphs)}, number of Data objects created for {label}: {len(graphs_data)}")
            
            results_df = cross_validate(graphs_data, label)
            all_results[label] = results_df  # Store results by label

    # Save all results to a single CSV file
    final_results_df = save_results_to_csv(all_results.values())
    
    # Plot combined results
    plot_combined_loss(all_results)
    plot_combined_r2(all_results)

    # Plot true vs. predicted for all labels
    plot_true_vs_predicted(labels)

if __name__ == "__main__":
    main()