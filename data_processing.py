"""
Module for processing graph data for the GNN_prediction project.
"""

import warnings
warnings.filterwarnings('ignore')

import pickle
import random
from tqdm import tqdm

class DataProcessor:
    """
    Class for processing graph data from various crystal systems.
    """

    @staticmethod
    def process_graphs():
        """
        Process graphs from different crystal systems.

        Returns:
            list: A list of processed graphs.
        """
        crystal_types = ['cubic', 'hexagonal', 'trigonal', 'tetragonal', 'orthorhombic', 'monoclinic', 'triclinic']
        feature_types = ['gnn']
        graphs = []

        for crystal_type in tqdm(crystal_types, total=len(crystal_types), desc="Reading graph files"):
            graphs_ = DataProcessor.read_graph_file(crystal_type, feature_types[0])
            random.shuffle(graphs_)
            graphs.extend(graphs_)

        print(f"Total number of graphs loaded: {len(graphs)}")
        random.shuffle(graphs)

        return graphs

    @staticmethod
    def read_graph_file(crystal_type, feature_type):
        """
        Read graph file for a specific crystal type and feature type.

        Args:
            crystal_type (str): Type of crystal system.
            feature_type (str): Type of feature.

        Returns:
            list: A list of graphs from the file.
        """
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
            return graphs
        except FileNotFoundError:
            print(f"File not found: {output_file}")
            return []

    @staticmethod
    def clean_graphs_for_y_label(values, label):
        """
        Clean graphs for a specific y-label.

        Args:
            values (list): List of values for the y-label.
            label (str): The y-label to clean for.

        Returns:
            list: Indices of selected graphs after cleaning.
        """
        # Implementation of cleaning logic for each label
        # This method should return the selected indices
        pass