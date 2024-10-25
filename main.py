from data_processing import DataProcessor
from graph_generation import GraphGenerator
from train_test import TrainTest

class GNNPrediction:
    def __init__(self):
        self.graphs = None
        self.data_list = None
        self.valid_y_labels = None

    def process_data(self):
        self.graphs = DataProcessor.process_graphs()

    def generate_graphs(self):
        self.data_list, self.valid_y_labels = GraphGenerator.convert_graphs_to_data_list(self.graphs)
        print(f"Number of Data objects created: {len(self.data_list)}")

    def train_and_evaluate(self):
        TrainTest.cross_validate(self.data_list, self.valid_y_labels)

    def run(self):
        self.process_data()
        self.generate_graphs()
        self.train_and_evaluate()

if __name__ == '__main__':
    gnn_prediction = GNNPrediction()
    gnn_prediction