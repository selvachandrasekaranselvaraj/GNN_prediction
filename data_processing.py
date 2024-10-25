import pickle
import random
from tqdm import tqdm

class DataProcessor:
    @staticmethod
    def process_graphs():
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