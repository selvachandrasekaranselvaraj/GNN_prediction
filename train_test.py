"""
Module for training and testing the GNN model.

This module contains the TrainTest class, which provides methods for training,
validating, and cross-validating a Graph Neural Network (GNN) model for predicting
material properties.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
from .model import GNN
from .utils import save_results_to_csv, plot_results

class TrainTest:
    """
    Class for training and testing the GNN model.
    """

    @staticmethod
    def train(model, train_loader, optimizer, criterion):
        """
        Train the model for one epoch.

        Args:
            model (GNN): The GNN model.
            train_loader (DataLoader): DataLoader for training data.
            optimizer (Optimizer): The optimizer for training.
            criterion (Loss): The loss function.

        Returns:
            tuple: A tuple containing the average training loss and R² score.
        """
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

    @staticmethod
    def validate(model, val_loader, criterion):
        """
        Validate the model.

        Args:
            model (GNN): The GNN model.
            val_loader (DataLoader): DataLoader for validation data.
            criterion (Loss): The loss function.

        Returns:
            tuple: A tuple containing the average validation loss, R² score, true values, and predicted values.
        """
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

    @staticmethod
    def cross_validate(data_list, valid_y_labels, epochs=1000, batch_size=256, learning_rate=1e-4, patience=50,
                       hidden_dim=200, num_layers=3, dropout=5e-5):
        """
        Perform cross-validation for each y-label.

        Args:
            data_list (list): List of PyTorch Geometric Data objects.
            valid_y_labels (list): List of valid y-labels.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            learning_rate (float): Learning rate for the optimizer.
            patience (int): Number of epochs to wait before early stopping.
            hidden_dim (int): Hidden dimension of the GNN model.
            num_layers (int): Number of GCN layers in the GNN model.
            dropout (float): Dropout rate for the GNN model.

        Returns:
            dict: A dictionary containing the results for each y-label.
        """
        all_results = {}

        for y_idx, y_label in enumerate(valid_y_labels):
            print(f"\nTraining model for {y_label}")
            
            y_data_list = [data for data in data_list]
            for data in y_data_list:
                data.y = data.y[y_idx]
            
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
                train_loss, train_r2 = TrainTest.train(model, train_loader, optimizer, criterion)
                val_loss, val_r2, _, _ = TrainTest.validate(model, val_loader, criterion)
                
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

            # Save the best model
            models_dir = 'trained_models'
            os.makedirs(models_dir, exist_ok=True)
            torch.save(best_model, f'{models_dir}/{y_label}.pth')
            print(f"Best model for {y_label} saved.")

            # Save and plot results
            results_df = pd.DataFrame({
                'Epoch': range(1, len(train_losses) + 1),
                'Train_Loss': train_losses,
                'Val_Loss': val_losses,
                'Train_R²': train_r2_scores,
                'Val_R²': val_r2_scores
            })
            all_results[y_label] = results_df

        return all_results