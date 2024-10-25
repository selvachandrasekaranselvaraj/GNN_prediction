import warnings
warnings.filterwarnings('ignore')
from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN
import numpy as np
from tqdm import tqdm
from typing import List, Dict
import random
import torch
import multiprocessing as mp
import pickle
import os

MAPI_KEY = "kqVFipfnaDNK96Vtu1KV1Ff3bfyR5LZ9"

def get_structures_from_mp(api_key, elements):
    with MPRester(api_key) as mpr:
        structures = mpr.materials.summary.search(elements=elements, num_elements=len(elements), fields=["structure", "formation_energy_per_atom"])
    return structures

def one_hot_encode(value: int, num_classes: int = 120) -> List[int]:
    encoding = [0] * num_classes
    encoding[value - 1] = 1  # Assuming atomic numbers start from 1
    return encoding

def structure_to_graph(structure_dict, output_file):
    structure = structure_dict['structure']
    y_values = []
    y_labels = ['volume', 
            'nsites',
            'density',  
            'bulk_modulus', 
            'shear_modulus', 
            'crystal_number', 
            'space_group_number', 
            'uncorrected_energy_per_atom', 
            'energy_per_atom', 
            'formation_energy_per_atom', 
            'energy_above_hull', 
            'is_stable', 
            'band_gap', 
            'cbm', 
            'vbm', 
            'efermi']
    for y_label in y_labels:
        if y_label == 'energy_per_atom' or y_label == 'uncorrected_energy_per_atom' :
            y_values.append(structure_dict[y_label]*structure_dict['nsites'])
        else:
            y_values.append(structure_dict[y_label])


    #structure, formation_energy = structure_dict['structure'], structure_dict['formation_energy_per_atom']
    #structure, formation_energy = structure_data
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
        'num_nodes': len(node_features),
        'node_features': node_features,
        'edge_indices': edge_indices,
        'edge_features': edge_features,
        'y_labels': y_values,
        #'formation_energy': formation_energy
    }

    # Append the graph to the output file
    with open(output_file, 'ab') as f:
        pickle.dump(graph, f)

    return True  # Return True to indicate successful processing

def process_structure(args):
    structure_dict, output_file, gnn_type, crystal_type, structure_number = args
    result = structure_to_graph(structure_dict, output_file)
    return 1 if result else 0  # Return 1 if successful, 0 otherwise

def generate_graphs(structures_dicts, gnn_type, crystal_type, max_workers, output_file):
    # Create the output file if it doesn't exist
    if not os.path.exists(output_file):
        open(output_file, 'wb').close()

    with mp.Pool(processes=max_workers) as pool:
        args = [(structure_dict, output_file, gnn_type, crystal_type, i) 
                for i, structure_dict in enumerate(structures_dicts)]
        
        results = list(tqdm(
            pool.imap(process_structure, args),
            total=len(structures_dicts),
            desc=f"Processing {crystal_type} - {gnn_type}"
        ))

    return results

def main():
    crystal_types = ['cubic', 'hexagonal', 'trigonal', 'tetragonal', 'orthorhombic', 'monoclinic', 'triclinic']
    #crystal_types = ['tetragonal', 'orthorhombic', 'monoclinic', 'triclinic']
    feature_types = ['gnn']

    for crystal_type in crystal_types:
        # Load the structures from the file
        with open(f'../structures/{crystal_type}_structures.pkl', 'rb') as f:
            structures_dicts = pickle.load(f)
        
        for feature_type in feature_types:
            output_file = f'graphs_{feature_type}_{crystal_type}.pkl'
            if os.path.exists(output_file):
                os.remove(output_file)
            results = generate_graphs(structures_dicts, feature_type, crystal_type, max_workers=32, output_file=output_file)

            successful_graphs = sum(results)
            print(f"Saved {successful_graphs} graphs for {crystal_type} - {feature_type} to {output_file}")

if __name__ == '__main__':
    mp.set_start_method('spawn')  # Required for CUDA to work with multiprocessing
    main()
