import torch
from torch_geometric.data import Data
from tqdm import tqdm

class GraphGenerator:
    @staticmethod
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

        data_list = [GraphGenerator.convert_graph_to_data(g, valid_y_indices) for g in tqdm(graphs, desc="Graphs2data")]

        return data_list, valid_y_labels

    @staticmethod
    def convert_graph_to_data(g, valid_y_indices):
        x = torch.tensor(g['node_features'], dtype=torch.float)
        edge_index = torch.tensor(g['edge_indices'], dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(g['edge_features'], dtype=torch.float).view(-1, 1)
        y_labels = [float(g['y_labels'][i]) for i in valid_y_indices]
        y = torch.tensor(y_labels, dtype=torch.float)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        return data