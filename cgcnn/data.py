import json 
import numpy as np 
from torch.utils.data import Dataset, DataLoader
from pymatgen.io.cif import CifParser
import os
import pandas as pd
from functools import cache
from pymatgen.core import Structure
import networkx as nx

class AtomDataInitializer:
    def __init__(self, path_to_atom_feature_vectors:str) -> None:
        with open (path_to_atom_feature_vectors, 'r') as file:
            data = json.load(file) 

        self.atom_embedding = {int(key): np.array(value, dtype=float) for key, value in data.items()}


class GaussianDistance():
    def __init__(self, dmin:int, dmax:int, step:int, var=None) -> None:
        pass

class EuclideanDisatance():
    pass


class CIFData(Dataset):
    def __init__(self, cif_directory, json_path, property_id_path,  max_neighbour=12, cutoff_radius=8, dmin=0, step=0.1, distance_type='gaussian') -> None:
        self.cif_directory =  cif_directory
        self.max_neighbour, self.cutoff_radius = max_neighbour, cutoff_radius

        if not os.path.exists(self.cif_directory):
            raise ValueError('CIF directory Not Found')
        if not os.path.exists(json_path):
            raise ValueError('json file Not Found')
        if not os.path.exists(property_id_path):
            raise ValueError('property ID file Not Found')
        
        self.prop_id_data = pd.read_csv(property_id_path)
        self.atomic_data = AtomDataInitializer(json_path)

        if distance_type == 'gaussian':
            self.dis = GaussianDistance(dmin=dmin, dmax=cutoff_radius, step=step)
        elif distance_type == 'euclidean':
            self.dis = EuclideanDisatance()
        else:
            raise ValueError('invalid distance type')
        
    def __len__(self):
        return len(self.prop_id_data)

    @cache
    def __getitem__(self, idx):
        cif_id, target_value = self.prop_id_data.loc[idx, ['cif_id', 'target_value']]
        cif_path = os.path.join(self.cif_directory, f'{cif_id}.cif')
        crystal = Structure.from_file(cif_path)
        all_neighbours = crystal.get_all_neighbors_py(self.cutoff_radius)

        






if __name__ == "__main__":
    cif = CIFData(cif_directory='/Users/sparsh/Desktop/College core/AI_DS_228/CGCNN-ME228/data/cif_files', json_path='/Users/sparsh/Desktop/College core/AI_DS_228/CGCNN-ME228/data/atom_init.json', property_id_path='')
    
        
        
    

