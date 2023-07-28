import pandas as pd
import torch
import torch_geometric
from torch_geometric.data import Dataset, Data
import numpy as np 
import os
from tqdm import tqdm
import networkx as nx
import json


print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Torch geometric version: {torch_geometric.__version__}")

import os.path as osp

import torch
from torch_geometric.data import Dataset, download_url

class MyOwnDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        return "forwarded_graph.csv"

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        return "forwarded_graph.pt"

    def download(self):
        # Download to `self.raw_dir`.
       pass

    def process(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        for index, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            row_obj = row
            # Get node features
            node_feats = self._get_node_features(row_obj)

            # Get adjacency info
            edge_index,_n = self._get_adjacency_info(row_obj)
            # Get labels info
            label = self._get_labels(row_obj)

            edge_indices = torch.tensor(edge_index)
            edge_indices = edge_indices.t().to(torch.long).view(2, -1)

            # Create data object
            data = Data(x=node_feats, 
                        edge_index=edge_indices,
                        y=label,
                        days = row["days"]
                        ) 
            
            torch.save(data, 
                os.path.join(self.processed_dir, f'data_{index}.pt'))

    def _get_adjacency_info(self, row):
        adg_list = row["adjacency_list"]
        dictionary = json.loads(adg_list)
        G = nx.from_dict_of_dicts(dictionary)

        adjacency_l = []
        for e in G.edges:
            adjacency_l.append((int(e[0]), int(e[1])))

        number_nodes = G.number_of_nodes()
    	
        return adjacency_l,number_nodes
    

    def _get_labels(self, row):
        label = row["patient_zero"]
        return torch.tensor(label, dtype=torch.int64)

    def _get_node_features(self, row):
        """ 
        This will return a matrix / 2d array of the shape
        [Number of Nodes, Node Feature size]
        """
        all_node_feats = []
        infected = json.loads(row["infected"])
        immune = json.loads(row["immune"])
        _var1,num_nodes = self._get_adjacency_info(row)
        nodes_idx = [range(0,num_nodes)]
        #not_affected = [x for x in nodes_idx if x not in infected and x not in immune]
        for idx in nodes_idx:
            node_feats = []
            if idx in immune:
                node_feats.append(0)
            elif idx in infected:
                node_feats.append(1)
            else:
                node_feats.append(2)

            # Append node features to matrix
            all_node_feats.append(node_feats)

        all_node_feats = np.asarray(all_node_feats)
        return torch.tensor(all_node_feats, dtype=torch.int)


    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data


dataset = MyOwnDataset("data/")
print(dataset[0].x)
print(dataset[0]. edge_index.t())
print(dataset[0].y)
print(dataset[0].days)
