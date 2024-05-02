import json
import warnings 
import numpy as np 
import torch
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
    
    def get_atomic_features(self, atom_type):
        return self.atom_embedding[atom_type]


class GaussianBasis():
    def __init__(self, dmin:int, dmax:int, step:int, var=None) -> None:
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var
    
    def expand_to_gaussian_basis(self, distances):
        distances = distances.reshape(distances.shape[0], distances.shape[1], 1)
        # we want to convert [[1,2,3], [4,5,6]] to [[[1],[2],[3]], [[4],[5],[6]]]
        return np.exp(-(distances - self.filter)**2 /
                      self.var**2)


class CIFData(Dataset):
    def __init__(self, cif_directory, json_path, property_id_path,  max_neighbour=12, cutoff_radius=8, dmin=0, step=0.1) -> None:
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

        self.basis = GaussianBasis(dmin=dmin, dmax=cutoff_radius, step=step)
        
    def __len__(self):
        return len(self.prop_id_data)

    @cache
    def __getitem__(self, idx):
        cif_id, property_value = self.prop_id_data.loc[idx, ['material_id', 'property_value']]
        cif_path = os.path.join(self.cif_directory, f'{cif_id}.cif')
        crystal = Structure.from_file(cif_path)
        all_neighbours = crystal.get_all_neighbors_py(self.cutoff_radius) #get_all_neighours_py is used because it returns a list[list[Periodic Neighbours]]
        all_neighbours = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_neighbours] #sort on the basis of distance from the atom
        neighbor_index = []
        neighbor_dist = []
        for atom_nbr in all_neighbours:
            temp = []
            temp_=[]
            if len(atom_nbr) < self.max_neighbour:
                warnings.warn('not enough numbers')
                for nbr in atom_nbr:
                    temp.append(nbr[2])
                    temp_.append(nbr[1])
                neighbor_index.append(temp + [0] * (self.max_neighbour - len(nbr)))
                neighbor_dist.append(temp_ + [self.cutoff_radius + 1.] * (self.max_neighbour - len(nbr)))
            else:
                for nbr in atom_nbr[:self.max_neighbour]:
                    temp.append(nbr[2]) # appends the index of the neighbor from 0 to 5 (index refers to atom in crystal)
                    temp_.append(nbr[1])
                neighbor_index.append(temp)
                neighbor_dist.append(temp_)

        neighbor_index,neighbor_dist = np.array(neighbor_index), np.array(neighbor_dist)
        print(neighbor_dist)
        neighbor_dist = self.basis.expand_to_gaussian_basis(neighbor_dist)
        #since the diff in distance value is so small we will use a gaussian basis to expand it 
        atomic_features = np.vstack([self.atomic_data.get_atomic_features(crystal[i].specie.number)
                              for i in range(len(crystal))])
        atomic_features = torch.Tensor(atomic_features)
        neighbor_dist = torch.Tensor(neighbor_dist) 
        neighbor_index = torch.Tensor(neighbor_index)
        target = torch.Tensor([float(property_value)])
        return (atomic_features, neighbor_dist, neighbor_index), target, cif_id


if __name__ == "__main__":
    cif = CIFData(cif_directory='/Users/sparsh/Desktop/College core/AI_DS_228/CGCNN-ME228/data/cif_files', json_path='/Users/sparsh/Desktop/College core/AI_DS_228/CGCNN-ME228/data/atom_init.json', property_id_path='/Users/sparsh/Desktop/College core/AI_DS_228/CGCNN-ME228/data/id_prop.csv')
    cif[1]
    
        
        
    

