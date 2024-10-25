import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
import matplotlib.pyplot as plt

class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super(GNN, self).__init__()
        self.convs = nn.ModuleList([GCNConv(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)])
        self.lin = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        mask = (x.sum(dim=-1) != 0)
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x[mask], batch[mask])
        x = self.lin(x)
        return x.squeeze()

class TrainTest:
    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def cross_validate(data_list, valid_y_labels, epochs=1000, batch_size=256, learning_rate=1e-4, patience=50,
                       hidden_dim=200, num_layers=3, dropout=5e-5):
        for y_idx, y_label in enumerate(valid_y_labels):
            print(f"\nTraining model for {y_label}")
            
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
                train_loss, train_r2 = TrainTest.train(model, train_loader, optimizer, criterion)
                val_loss, val_r2 = TrainTest.validate(model, val_loader, criterion)
                
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

            models_dir = f'trained_models'
            os.makedirs(models_dir, exist_ok=True)
            torch.save(best_model, f'./{models_dir}/{y_label}.pth')
            print(f"Best model for {y_label} saved.")

            output_dir=f'results_plots'
            os.makedirs(output_dir, exist_ok=True)
            results_df = TrainTest.save_results_to_csv(train_losses, val_losses, train_r2_scores, val_r2_scores, filename=f'./{output_dir}/{y_label}.csv')

            TrainTest.plot_and_save_results(results_df, y_label, output_dir=f'results_plots')