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
from tqdm import tqdm
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import multiprocessing

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
    return total_loss / len(val_loader), r2

def save_results_to_csv(train_losses, val_losses, train_r2_scores, val_r2_scores, filename='training_results.csv'):
    results_df = pd.DataFrame({
        'Epoch': range(1, len(train_losses) + 1),
        'Train_Loss': train_losses,
        'Val_Loss': val_losses,
        'Train_R²': train_r2_scores,
        'Val_R²': val_r2_scores
    })
    results_df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")
    return results_df

def plot_and_save_results(results_df, y_label, output_dir='results_plots'):
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

def cross_validate(data_list, valid_y_labels, epochs=1000, batch_size=256, learning_rate=1e-4, patience=50,
                   hidden_dim=200, num_layers=3, dropout=5e-5):
    for y_idx, y_label in enumerate(valid_y_labels):
        print(f"\nTraining model for {y_label}")
        
        # Prepare data for this y_label
        y_data_list = [Data(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, y=data.y[y_idx]) for data in data_list]
        
        train_data, val_data = train_test_split(y_data_list, test_size=0.2, random_state=42)
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

        for epoch in range(epochs):
            train_loss, train_r2 = train(model, train_loader, optimizer, criterion)
            val_loss, val_r2 = validate(model, val_loader, criterion)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_r2_scores.append(train_r2)
            val_r2_scores.append(val_r2)
            
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train R2: {train_r2:.4f}, Val R2: {val_r2:.4f}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve == patience:
                print(f"Early stopping triggered.")
                break

        # Save the best model for this y_label
        models_dir = f'trained_models'
        os.makedirs(models_dir, exist_ok=True)
        torch.save(best_model, f'./{models_dir}/{y_label}.pth')
        print(f"Best model for {y_label} saved.")

        # Save results to CSV
        output_dir=f'results_plots'
        os.makedirs(output_dir, exist_ok=True)
        results_df = save_results_to_csv(train_losses, val_losses, train_r2_scores, val_r2_scores, filename=f'./{output_dir}/{y_label}.csv')

        # Plot results
        plot_and_save_results(results_df, y_label, output_dir=f'results_plots')

def convert_graph_to_data(g, valid_y_indices):
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
    y_labels = [float(g['y_labels'][i]) for i in valid_y_indices]
    y = torch.tensor(y_labels, dtype=torch.float)

    # Create Data object
    y = torch.tensor(y_labels, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    return data

def convert_graphs_to_data_list(graphs):
    y_label_names = ['density', 'bulk_modulus', 'crystal_number', 'space_group_number',
                     'energy_per_atom', 'formation_energy_per_atom', 'energy_above_hull',
                     'is_stable', 'band_gap']

    valid_y_indices = []
    for i, label_name in enumerate(y_label_names):
        if all(g['y_labels'][i] is not None for g in graphs):
            valid_y_indices.append(i)

    valid_y_labels = [y_label_names[i] for i in valid_y_indices]
    print(f"Valid y_labels: {valid_y_labels}")

    data_list = [convert_graph_to_data(g, valid_y_indices) for g in tqdm(graphs, desc="Graphs2data")]

    return data_list, valid_y_labels

def process_graphs():
    crystal_types = ['cubic', 'hexagonal', 'trigonal', 'tetragonal', 'orthorhombic', 'monoclinic', 'triclinic']
    feature_types = ['gnn']
    graphs = []

    for crystal_type in tqdm(crystal_types, total=len(crystal_types), desc="Reading graph files"):
        graphs_ = read_graph_file(crystal_type, feature_types[0])
        random.shuffle(graphs_)
        graphs.extend(graphs_)

    print(f"Total number of graphs loaded: {len(graphs)}")
    random.shuffle(graphs)

    return graphs


def read_graph_file(crystal_type, feature_type):
    output_file = f'../../graphs/graphs_{feature_type}_{crystal_type}.pkl'
    graphs = []
    try:
        with open(output_file, 'rb') as f:
            while True:
                try:
                    graph = pickle.load(f)
                    graphs.append(graph)
                except EOFError:
                    break
        #print(f"File loaded: {output_file}")
        return graphs
    except FileNotFoundError:
        print(f"File not found: {output_file}")
        return []


def main():
    graphs = process_graphs()
    data_list, valid_y_labels = convert_graphs_to_data_list(graphs)

    print(f"Number of Data objects created: {len(data_list)}")

    # Perform cross-validation
    cross_validate(data_list, valid_y_labels)


if __name__ == '__main__':
    main()
