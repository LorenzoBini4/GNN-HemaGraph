import subprocess  # Import the subprocess module
import os
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.loader import ClusterData, ClusterLoader
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import NeighborSampler
import dgl
import torch.nn as nn
import torch.optim as optim
from dgl.nn import GraphConv
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv
from torch_geometric.data import DataLoader
from torch.distributions import Normal
from torch.distributions.transforms import AffineTransform
from torch_geometric.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from dgl.data import CoraGraphDataset
from sklearn.neighbors import NearestNeighbors
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch_geometric.nn import GraphConv
from sklearn.metrics import f1_score
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import numpy as np
import random
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split


# Load the list of graphs
dataset = torch.load('/home/users/b/bini/gnn/a_cells_sup/Graph/flat_A7.pt')

# Print the number of graphs in the final list
print(f"Number of initial graphs in the list: {len(dataset)}")

# Define the split ratios
train_ratio = 0.8
val_test_ratio = 0.1 # 0.05 for validation and 0.05 for test

# Define a function to compute class distribution percentages
def compute_class_distribution_percentage(mask, labels):
    class_count = torch.zeros(labels.max().item() + 1)
    total_nodes = 0
    for i, label in enumerate(labels):
        if mask[i]:
            class_count[label.item()] += 1
            total_nodes += 1
    return (class_count / total_nodes) * 100  # Calculate percentage

# Split each graph and print class distribution percentages
for i, data in enumerate(dataset):
    num_nodes = data.x.size(0)
    num_classes = int(data.y.max()) + 1
    
    # Initialize masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    for c in range(num_classes):
        # Get nodes of class c
        nodes_of_class_c = (data.y == c).nonzero(as_tuple=True)[0]
        
        # Shuffle the nodes
        nodes_of_class_c = nodes_of_class_c[torch.randperm(nodes_of_class_c.size(0))]
        
        # Split nodes into train, val, and test
        num_train_nodes = int(train_ratio * len(nodes_of_class_c))
        num_val_test_nodes = int(val_test_ratio * len(nodes_of_class_c))
        
        train_nodes = nodes_of_class_c[:num_train_nodes]
        val_nodes = nodes_of_class_c[num_train_nodes:num_train_nodes + num_val_test_nodes]
        test_nodes = nodes_of_class_c[num_train_nodes + num_val_test_nodes:]
        
        # Update masks
        train_mask[train_nodes] = True
        val_mask[val_nodes] = True
        test_mask[test_nodes] = True
    
    # Assign masks to the data object
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    # Compute and print class distribution percentages
    train_percentage = compute_class_distribution_percentage(train_mask, data.y)
    val_percentage = compute_class_distribution_percentage(val_mask, data.y)
    test_percentage = compute_class_distribution_percentage(test_mask, data.y)
    
    print(f"Graph {i + 1}:")
    print("Train Mask - Class Distribution Percentage:")
    for label, percentage in enumerate(train_percentage):
        print(f"  Class {label}: {percentage:.2f}%")
    
    print("\nValidation Mask - Class Distribution Percentage:")
    for label, percentage in enumerate(val_percentage):
        print(f"  Class {label}: {percentage:.2f}%")
    
    print("\nTest Mask - Class Distribution Percentage:")
    for label, percentage in enumerate(test_percentage):
        print(f"  Class {label}: {percentage:.2f}%")
    
    print("\n" + "-"*50 + "\n")  # Separator for better readability

# Save the modified dataset
torch.save(dataset, 'masked_graph.pt')

# Load the masked dataset
dataset = torch.load('masked_graph.pt')

# Initialize an empty list to store class weights for each graph
class_weights_tensor = []

# Iterate over each graph in the dataset to compute class weights
for i, data in enumerate(dataset):
    # Calculate class frequencies for the current graph
    num_classes = int(data.y.max()) + 1
    class_counts = torch.zeros(num_classes)
    
    for c in range(num_classes):
        class_counts[c] += (data.y == c).sum()

    # Compute inverse frequencies for class weights
    total_samples = sum(class_counts)
    class_weights = total_samples / class_counts

    # Normalize the class weights
    class_weights = class_weights / class_weights.sum()
    
    # Adjust the last element of class_weights
    #last_weight = class_weights[-1]
    #second_last_weight = class_weights[-2]
    #class_weights[-1] = (last_weight + second_last_weight) / 2

    # Append class weights to the list
    class_weights_tensor.append(class_weights)

    # Print class weights for the current graph (optional)
    print(f"Graph {i + 1} Class Weights:", class_weights)
    print("-" * 50)

# Now, class_weights_tensor contains 30 different class_weights tensors
# Each element of class_weights_tensor corresponds to a graph's class weights
print(class_weights_tensor)

