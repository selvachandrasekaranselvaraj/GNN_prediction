"""
This module initializes the GNN_prediction package.

It imports and exposes the main components of the package for easy access:
    - DataProcessor: For processing input data.
    - GraphGenerator: For generating graph representations of the data.
    - GNN: The Graph Neural Network model.
    - TrainTest: For training and testing the model.
    - plot_results and save_results_to_csv: Utility functions for handling results.

This module allows users to easily import the main components of the package using:
    from gnn_prediction import DataProcessor, GraphGenerator, GNN, TrainTest, plot_results, save_results_to_csv
"""

from .data_processing import DataProcessor
from .graph_generation import GraphGenerator
from .model import GNN
from .train_test import TrainTest
from .utils import plot_results, save_results_to_csv