import FlowCal
import torch
import numpy as np
import itertools
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data


# Selected columns
selected_columns = [ 'FS INT',  'SS INT', 'FL1 INT_CD14-FITC', 'FL2 INT_CD19-PE', 'FL3 INT_CD13-ECD', 'FL4 INT_CD33-PC5.5', 'FL5 INT_CD34-PC7', 'FL6 INT_CD117-APC', 'FL7 INT_CD7-APC700', 'FL8 INT_CD16-APC750', 'FL9 INT_HLA-PB', 'FL10 INT_CD45-KO']

# Function to read FCS files and extract specified columns
def read_and_extract_columns(file_path, selected_columns):
    fcs_data = FlowCal.io.FCSData(file_path)
    selected_data = fcs_data[:, selected_columns].view()
    return selected_data

# Function to create k-NN graph with k=10 using the specified method
def create_knn_graph(data, k=7):
    # Calculate pairwise distances between data points
    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(data)
    distances, indices = nbrs.kneighbors(data)
    
    # Create edge_index for the k-NN graph
    src = np.repeat(np.arange(data.shape[0]), k)
    dst = indices.flatten()
    
    # Convert numpy arrays to PyTorch tensors
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    
    return edge_index


# List of FCS file paths
file_paths = ['insert/file_paths/.fcs'
]

# Read and process each FCS file
all_graphs = []
for file_path in file_paths:
    selected_data = read_and_extract_columns(file_path, selected_columns)
    edge_index = create_knn_graph(selected_data, k=7)
    
    # Normalized_data
    normalized_data = selected_data

    # Convert to PyG Data format
    graph_data = Data(x=torch.tensor(normalized_data, dtype=torch.float32),
                      edge_index=edge_index, edge_attr=None)
    all_graphs.append(graph_data)

# Save all PyG graphs to a file
torch.save(all_graphs, 'flat_graph.pt')

