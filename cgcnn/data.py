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
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


def get_train_val_test_loader(dataset, collate_fn=default_collate, batch_size=64,
                              train_ratio=None, val_ratio=0.1, test_ratio=0.1,
                              return_test=True, num_workers=1, **kwargs):
    """
    Get train, validation, and test loaders from a dataset.

    Parameters:
    -----------
    dataset : Dataset
        The dataset from which to create loaders.
    collate_fn : callable, optional (default=torch.utils.data.dataloader.default_collate)
        Function to collate samples into batches.
    batch_size : int, optional (default=64)
        Batch size for the loaders.
    train_ratio : float, optional (default=None)
        Ratio of the dataset to use for training. If None, computed from val_ratio and test_ratio.
    val_ratio : float, optional (default=0.1)
        Ratio of the dataset to use for validation.
    test_ratio : float, optional (default=0.1)
        Ratio of the dataset to use for testing.
    return_test : bool, optional (default=True)
        Whether to return a test loader.
    num_workers : int, optional (default=1)
        Number of subprocesses to use for data loading.
    **kwargs : dict, optional
        Additional keyword arguments.

    Returns:
    --------
    train_loader : torch.utils.data.DataLoader
        DataLoader for training data.
    val_loader : torch.utils.data.DataLoader
        DataLoader for validation data.
    test_loader : torch.utils.data.DataLoader, optional
        DataLoader for test data, returned only if return_test is True.
    """
    total_size = len(dataset)  # Total size of the dataset

    # Compute sizes of train, validation, and test sets based on ratios or explicit sizes
    if kwargs.get('train_size') is None:
        if train_ratio is None:
            assert val_ratio + test_ratio < 1
            train_ratio = 1 - val_ratio - test_ratio
            print(f'[Warning] train_ratio is None, using 1 - val_ratio - '
                  f'test_ratio = {train_ratio} as training data.')
        else:
            assert train_ratio + val_ratio + test_ratio <= 1
    indices = list(range(total_size))  # List of indices of the dataset

    # Compute sizes of train, validation, and test sets
    if kwargs.get('train_size'):
        train_size = kwargs['train_size']
    else:
        train_size = int(train_ratio * total_size)
    if kwargs.get('test_size'):
        test_size = kwargs['test_size']
    else:
        test_size = int(test_ratio * total_size)
    if kwargs.get('val_size'):
        valid_size = kwargs['val_size']
    else:
        valid_size = int(val_ratio * total_size)

    # Create samplers for train, validation, and test sets
    train_sampler = SubsetRandomSampler(indices[:train_size])
    val_sampler = SubsetRandomSampler(
        indices[-(valid_size + test_size):-test_size])
    if return_test:
        test_sampler = SubsetRandomSampler(indices[-test_size:])

    # Create DataLoader instances for train, validation, and test sets
    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              collate_fn=collate_fn)
    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=val_sampler,
                            num_workers=num_workers,
                            collate_fn=collate_fn)
    if return_test:
        test_loader = DataLoader(dataset, batch_size=batch_size,
                                 sampler=test_sampler,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn)

    # Return train, validation, and test loaders
    if return_test:
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader

    
def collate_pool(dataset_list):
    """
    Collate a list of samples into batches.

    Parameters:
    -----------
    dataset_list : list
        List of samples, where each sample is a tuple containing atom features,
        neighbor features, neighbor indices, target, and CIF ID.

    Returns:
    --------
    tuple
        A tuple containing the batched atom features, neighbor features,
        neighbor indices, crystal atom indices, batch target, and CIF IDs.
    """
    # Initialize lists to store batched data
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    crystal_atom_idx, batch_target = [], []
    batch_cif_ids = []
    base_idx = 0  # Initialize base index for crystal atom indices

    # Iterate over each sample in the dataset list
    for i, ((atom_fea, nbr_fea, nbr_fea_idx), target, cif_id) in enumerate(dataset_list):
        n_i = atom_fea.shape[0]  # Number of atoms for this crystal
        # Append atom features, neighbor features, and neighbor indices to the batch lists
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx + base_idx)
        # Create new indices for crystal atom indices and append to the list
        new_idx = torch.LongTensor(np.arange(n_i) + base_idx)
        crystal_atom_idx.append(new_idx)
        # Append target and CIF ID to the batch lists
        batch_target.append(target)
        batch_cif_ids.append(cif_id)
        base_idx += n_i  # Update base index for the next crystal

    # Concatenate batched atom features, neighbor features, and neighbor indices
    # Convert crystal atom indices to a tensor and stack batch target
    return (torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            torch.stack(crystal_atom_idx),
            torch.stack(batch_target, dim=0),
            batch_cif_ids)  # Return batched data as a tuple


class AtomDataInitializer:
    def __init__(self, path_to_atom_feature_vectors:str) -> None:
        """
        Initialize AtomDataInitializer object.

        Parameters:
        -----------
        path_to_atom_feature_vectors : str
            Path to the JSON file containing atom feature vectors.
        """
        # Load atom feature vectors from JSON file
        with open(path_to_atom_feature_vectors, 'r') as file:
            data = json.load(file) 

        # Create a dictionary to store atom embeddings
        # Convert keys to integers and values to numpy arrays
        self.atom_embedding = {int(key): np.array(value, dtype=float) for key, value in data.items()}
    
    def get_atomic_features(self, atom_type):
        """
        Get atomic features for a given atom type.

        Parameters:
        -----------
        atom_type : int
            Atom type for which to retrieve atomic features.

        Returns:
        --------
        numpy.ndarray
            Array containing atomic features for the specified atom type.
        """
        # Retrieve atomic features from the atom_embedding dictionary
        return self.atom_embedding[atom_type]



class GaussianBasis():
    def __init__(self, dmin:int, dmax:int, step:int, var=None) -> None:
        """
        Initialize GaussianBasis object.

        Parameters:
        -----------
        dmin : int
            Minimum distance value for Gaussian filter.
        dmax : int
            Maximum distance value for Gaussian filter.
        step : int
            Step size between distance values.
        var : int, optional (default=None)
            Variance of the Gaussian basis function. If None, set to step.
        """
        # Generate the filter array using specified range and step size
        self.filter = np.arange(dmin, dmax+step, step)
        # Set variance of Gaussian basis function
        if var is None:
            var = step
        self.var = var
    
    def expand_to_gaussian_basis(self, distances):
        """
        Expand distances to Gaussian basis representation.

        Parameters:
        -----------
        distances : numpy.ndarray
            Array of distances.

        Returns:
        --------
        numpy.ndarray
            Expanded distances in Gaussian basis representation.
        """
        # Reshape distances array to have an additional axis
        distances = distances.reshape(distances.shape[0], distances.shape[1], 1)
        # Compute Gaussian basis representation
        gaussian_basis = np.exp(-(distances - self.filter)**2 / self.var**2)
        return gaussian_basis



class CIFData(Dataset):
    def __init__(self, cif_directory, json_path, property_id_path,  max_neighbour=12, cutoff_radius=8, dmin=0, step=0.1) -> None:
        """
        Initialize CIFData dataset.

        Parameters:
        -----------
        cif_directory : str
            Path to the directory containing CIF files.
        json_path : str
            Path to the JSON file containing atom feature vectors.
        property_id_path : str
            Path to the CSV file containing property IDs.
        max_neighbour : int, optional (default=12)
            Maximum number of neighbors.
        cutoff_radius : int, optional (default=8)
            Cutoff radius for neighbor search.
        dmin : int, optional (default=0)
            Minimum distance value for Gaussian basis.
        step : float, optional (default=0.1)
            Step size between distance values for Gaussian basis.
        """
        # Initialize dataset parameters
        self.cif_directory =  cif_directory
        self.max_neighbour, self.cutoff_radius = max_neighbour, cutoff_radius

        # Check if directory and files exist
        if not os.path.exists(self.cif_directory):
            raise ValueError('CIF directory Not Found')
        if not os.path.exists(json_path):
            raise ValueError('json file Not Found')
        if not os.path.exists(property_id_path):
            raise ValueError('property ID file Not Found')
        
        # Load property ID data and atom feature data
        self.prop_id_data = pd.read_csv(property_id_path)
        self.atomic_data = AtomDataInitializer(json_path)

        # Initialize Gaussian basis
        self.basis = GaussianBasis(dmin=dmin, dmax=cutoff_radius, step=step)
        
    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
        --------
        int
            Length of the dataset.
        """
        return len(self.prop_id_data)

    @cache
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Parameters:
        -----------
        idx : int
            Index of the sample.

        Returns:
        --------
        tuple
            Tuple containing atomic features, neighbor distances, neighbor indices, target, and CIF ID.
        """
        # Get CIF ID and property value
        cif_id, property_value = self.prop_id_data.loc[idx, ['material_id', 'property_value']]
        # Load CIF file
        cif_path = os.path.join(self.cif_directory, f'{cif_id}.cif')
        crystal = Structure.from_file(cif_path)
        # Get all neighbors within cutoff radius
        all_neighbours = crystal.get_all_neighbors_py(self.cutoff_radius)
        # Sort neighbors based on distance from the atom
        all_neighbours = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_neighbours]
        # Initialize lists to store neighbor indices and distances
        neighbor_index = []
        neighbor_dist = []
        # Iterate over each atom and its neighbors
        for atom_nbr in all_neighbours:
            temp = []
            temp_=[]
            # Check if number of neighbors is less than maximum allowed
            if len(atom_nbr) < self.max_neighbour:
                warnings.warn('not enough numbers')
                # If less, pad with zeros
                for nbr in atom_nbr:
                    temp.append(nbr[2])
                    temp_.append(nbr[1])
                neighbor_index.append(temp + [0] * (self.max_neighbour - len(nbr)))
                neighbor_dist.append(temp_ + [self.cutoff_radius + 1.] * (self.max_neighbour - len(nbr)))
            else:
                # If not, take only the first `max_neighbour` neighbors
                for nbr in atom_nbr[:self.max_neighbour]:
                    temp.append(nbr[2])
                    temp_.append(nbr[1])
                neighbor_index.append(temp)
                neighbor_dist.append(temp_)
        # Convert lists to numpy arrays
        neighbor_index, neighbor_dist = np.array(neighbor_index), np.array(neighbor_dist)
        # Expand distances to Gaussian basis representation
        neighbor_dist = self.basis.expand_to_gaussian_basis(neighbor_dist)
        # Get atomic features for each atom in the crystal
        atomic_features = np.vstack([self.atomic_data.get_atomic_features(crystal[i].specie.number)
                              for i in range(len(crystal))])
        atomic_features = torch.Tensor(atomic_features)
        neighbor_dist = torch.Tensor(neighbor_dist) 
        neighbor_index = torch.Tensor(neighbor_index)
        target = torch.Tensor([float(property_value)])
        # Return sample data as a tuple
        return (atomic_features, neighbor_dist, neighbor_index), target, cif_id


if __name__ == "__main__":
    cif = CIFData(cif_directory='/Users/sparsh/Desktop/College core/AI_DS_228/CGCNN-ME228/data/cif_files', json_path='/Users/sparsh/Desktop/College core/AI_DS_228/CGCNN-ME228/data/atom_init.json', property_id_path='/Users/sparsh/Desktop/College core/AI_DS_228/CGCNN-ME228/data/id_prop.csv')
    cif[1]
    
        
        
    

