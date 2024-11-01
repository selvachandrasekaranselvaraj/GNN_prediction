import warnings
warnings.filterwarnings('ignore')
from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN
import numpy as np
from tqdm import tqdm
from typing import List, Dict
import random
import torch
import pickle
import os

def one_hot_encode(value: int, num_classes: int = 120) -> List[int]:
    encoding = [0] * num_classes
    encoding[value - 1] = 1  # Assuming atomic numbers start from 1
    return encoding

def structure_to_graph(structure_dict, output_file, i):
    structure = structure_dict['structure']
    y_label_names = ['volume', 'nsites', 'density', 'bulk_modulus', 'shear_modulus',
                     'crystal_number', 'space_group_number', 'uncorrected_energy_per_atom',
                     'energy_per_atom', 'formation_energy_per_atom', 'energy_above_hull',
                     'is_stable', 'band_gap', 'cbm', 'vbm', 'efermi']
    plot_labels = ['V', 'n', 'ρ', 'B voigt', 'B reuss', 'B vrh', 'G voigt', 'G reuss', 'G vrh', 'CN', 'SGN', 'UE', 'E', 'FE', 'EAH', 'is_S', 'EG', 'CBM', 'VBM', 'Ef']
    y_values = []
    all_y_labels = []

    for label in y_label_names:
        value = structure_dict.get(label)
        if label in ['bulk_modulus', 'shear_modulus']:
            if isinstance(value, dict):
                for key in ['voigt', 'reuss', 'vrh']:
                    y_values.append(value[key])
            else:
                for key in ['voigt', 'reuss', 'vrh']:
                    y_values.append(None)        
        else:
            if label == 'volume':
               value /= structure_dict.get('nsites')
          
            y_values.append(value)
            #all_y_labels.append(label)
    cutoff = 8.0
    max_num_nbr = 12
    
    node_features = [one_hot_encode(site.specie.number) for site in structure]

    cnn = CrystalNN(search_cutoff=cutoff)
    all_nn_info = []
    
    for i in range(len(structure)):
        try:
            nn_info = cnn.get_nn_info(structure, i)
            nn_info = sorted(nn_info, key=lambda x: x['weight'])[:max_num_nbr]
            all_nn_info.append(nn_info)
        except:
            return None
    
    edge_indices = []
    edge_features = []
    
    for idx, nbrs in enumerate(all_nn_info):
        for nbr in nbrs:
            j = nbr['site_index']
            dist = structure.get_distance(idx, j)
            edge_indices.append((idx, j))
            edge_features.append([dist])
    
    graph = {
        'node_features': node_features,
        'edge_indices': edge_indices,
        'edge_features': edge_features,
        'y_values': y_values
    }

    # Append the graph to the output file
    with open(output_file, 'ab') as f:
        pickle.dump(graph, f)

    return True  # Return True to indicate successful processing

def main():
    crystal_types = ['hexagonal', 'trigonal', 'tetragonal', 'orthorhombic', 'monoclinic', 'triclinic']
    feature_types = ['gnn']

    for crystal_type in crystal_types:
        # Load the structures from the file
        with open(f'../structures/{crystal_type}_structures.pkl', 'rb') as f:
            structures_dicts = pickle.load(f)

        for feature_type in feature_types:
            output_file = f'graphs_{feature_type}_{crystal_type}.pkl'
            if os.path.exists(output_file):
                os.remove(output_file)

            successful_graphs = 0
            
            # Process each structure sequentially
            for i, structure_dict in enumerate(tqdm(structures_dicts, desc=f"Processing {crystal_type} - {feature_type}")):
                result = structure_to_graph(structure_dict, output_file, i)
                if result:
                    successful_graphs += 1
                    
            print(f"Processed {i + 1} structures for {crystal_type} - {feature_type}")
            print(f"Saved {successful_graphs} graphs for {crystal_type} - {feature_type} to {output_file}")

if __name__ == '__main__':
    main()

