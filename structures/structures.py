"""
This module handles the retrieval and processing of crystal structures from the Materials Project database.

It uses the Materials Project API to download structures of various crystal systems and converts them into a dictionary format.

Functions:
    structures_from_mp: Retrieve structures from the Materials Project database for a given crystal system.
    structure_to_dict: Convert a structure object to a dictionary format.
    main: The main function to download and process structures for all crystal systems.

Constants:
    MAPI_KEY: The Materials Project API key for authentication.

This module is typically run as a standalone script to create the initial dataset for the GNN_prediction project.
"""

from pymatgen.ext.matproj import MPRester
import random
import pickle
import time
import json

print("Starting")

MAPI_KEY = "...."

def structures_from_mp(api_key, crystal):
    with MPRester(api_key) as mpr:
        structures = mpr.materials.summary.search(
            crystal_system=crystal,
            #energy_above_hull=(0.0, 1.011),
            all_fields=True
        )
        available_fields = mpr.materials.summary.available_fields
        return structures, available_fields

def structure_to_dict(structure, crystal_number, available_fields):
    data = {}
    for field in available_fields:
        if hasattr(structure, field):
            if field == 'symmetry':
                data['crystal_system'] = getattr(structure, field).crystal_system
                data['crystal_number'] = crystal_number
                data['space_group'] = getattr(structure, field).symbol
                data['space_group_number'] = getattr(structure, field).number
                data['point_group'] = getattr(structure, field).point_group
            else:
                data[field] = getattr(structure, field)
                
    return data

def main():
    crystals = [
        'Cubic',
        'Hexagonal',
        'Trigonal',
        'Tetragonal',
        'Orthorhombic',
        'Monoclinic',
        'Triclinic',
    ]

    for c_no, crystal in enumerate(crystals, start=1):
        print(f"Download started for {crystal.lower()} structures...")
        structures, available_fields = structures_from_mp(MAPI_KEY, crystal)

        random.shuffle(structures)

        print(f"Downloaded {len(structures)} {crystal.lower()} structures")

        structure_dicts = [structure_to_dict(s, c_no, available_fields) for s in structures]

        print(f"Converted {crystal.lower()} {len(structure_dicts)} structures to dictionaries")

        filename = f"{crystal.lower()}_structures.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(structure_dicts, f)

        print(f"Saved {crystal.lower()} {len(structure_dicts)} structures to {filename}")

if __name__ == "__main__":
    main()
