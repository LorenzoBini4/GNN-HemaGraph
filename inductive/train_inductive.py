"""
This script trains and evaluates a Graph Attention Network (GAT) model on a multiclass classification task.
The GAT model is implemented using PyTorch and PyTorch Geometric library.
The script performs the following steps:
1. Set random seeds for reproducibility.
2.Check if a GPU is available and print GPU memory information.
3. Load the dataset and calculate the count of samples in each class.
4. Define the GAT model architecture.
5. Implement training and testing functions for the GAT model.
6. Perform k-fold cross-validation for model evaluation.
7. Save the predicted labels for each patient in a separate CSV file.
8. Compute and print evaluation metrics (confusion matrix, precision, recall, accuracy, F1 score) for each patient.
9. Calculate the ratio of correct predictions for each label across all patients.

Parameters:
- max_num_epochs: Maximum number of training epochs for the GAT model.
- batch_size: Batch size for training and testing.
- start_lr: Initial learning rate for the optimizer.
- num_repetitions: Number of repetitions for k-fold cross-validation.
- all_std: Boolean flag indicating whether to compute standard deviation of F1 scores across repetitions.

Returns:
- patient_dict: A dictionary containing F1 scores, predicted labels, and true labels for each patient.
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import itertools
import torch
from torch_geometric.data import Data, Dataset
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
import os
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import random
from utils.mask import class_weights_tensor
import torch.nn as nn

# Set seeds for reproducibility
seed_value = 77  # You can use any integer value

# Set seed for random module
random.seed(seed_value)

# Set seed for NumPy
np.random.seed(seed_value)

# Set seed for PyTorch
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

# Set seed for CUDA operations (if available)
#torch.backends.cudnn.deterministic = True 
#torch.backends.cudnn.benchmark = False

PRINT_MEMORY = False
device_string = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_string)
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:21"

# Check if a GPU is available
if torch.cuda.is_available():
    # Get the current GPU device
    device = torch.cuda.current_device()

    # Get the GPU's memory usage in bytes
    memory_allocated = torch.cuda.memory_allocated(device)
    memory_cached = torch.cuda.memory_cached(device)

    # Convert bytes to a more human-readable format (e.g., megabytes or gigabytes)
    memory_allocated_mb = memory_allocated / 1024**2  # Megabytes
    memory_cached_mb = memory_cached / 1024**2  # Megabytes

    print(f"GPU Memory Allocated: {memory_allocated_mb:.2f} MB")
    print(f"GPU Memory Cached: {memory_cached_mb:.2f} MB")
else:
    print("No GPU available.")


### graph with k=7 from kNN, and nodes have been min-max normalized ###
data_FC = torch.load('flat_graph.pt')

label0_count=[]

for j in range(30):
        df = pd.read_csv(f"Data_original/Case_{j+1}.csv")
        label0_count.append(len(df))
#print(label0_count)

# Convert class weights to a PyTorch tensor
class_weights_tensor = [weights.to(device) for weights in class_weights_tensor]

class MyGraphDataset(Dataset):
    def __init__(self,  num_samples,transform=None, pre_transform=None):
        super(MyGraphDataset, self).__init__(transform, pre_transform)
        self.num_samples = num_samples
        self.data_list = torch.load('flat_graph.pt')
        self.class_weights = class_weights_tensor  # Add the class_weights attribute

    def len(self):
        return self.num_samples

    def get(self, idx):
        return self.data_list[idx]

class HemaGraph(torch.nn.Module):
    def __init__(self):
        super(HemaGraph, self).__init__()
        self.hid = 64
        self.in_head = 8
        self.out_head = 8

        # First GAT layer
        self.conv1 = GATConv(12, self.hid, heads=self.in_head, dropout=0.4)

        # Second GAT layer
        self.conv2 = GATConv(self.hid * self.in_head, self.hid, heads=self.in_head, dropout=0.4)

        # Third GAT layer
        self.conv3 = GATConv(self.hid * self.in_head, 5, concat=False,
                             heads=self.out_head, dropout=0.4)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=0.4, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = F.dropout(x, p=0.4, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = F.dropout(x, p=0.4, training=self.training)
        x = self.conv3(x, edge_index)

        return F.log_softmax(x, dim=1)

    def __repr__(self):
        return self.__class__.__name__

# One training epoch for GNN model.
def train(train_loader, model, optimizer, device, class_weights_tensor):
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        # Apply combined mask
        current_weights = class_weights_tensor[batch_idx]
        criterion = nn.NLLLoss(weight=current_weights)
        #criterion = nn.CrossEntropyLoss(weight=current_weights)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
    #return output

# Get acc. of GNN model.
def test(loader, model, device):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        output = model(data)
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()/len(pred)
    return correct / len(loader.dataset)



def gnn_evaluation(gnn, max_num_epochs=1000, batch_size=128, start_lr=0.01, min_lr=0.000001, factor=0.5, patience=5,
                   num_repetitions=10, all_std=True):
    dataset = MyGraphDataset(num_samples=len(torch.load('flat_graph.pt'))).shuffle()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    best_model_state_dict = None
    # Add these lines before the cross-validation loop
    best_test_indices = None
    best_f1_score = 0.0
    patient_dict=dict()
    for i in range(num_repetitions):
        kf = KFold(n_splits=7, shuffle=True)
        dataset.shuffle()

        f1_scores = []
        acc_scores = []
        all_preds = []
        all_labels = []

        for train_index, test_index in kf.split(list(range(len(dataset)))):
            train_index, val_index = train_test_split(train_index, test_size=0.1)

            train_dataset = dataset[train_index.tolist()]
            val_dataset = dataset[val_index.tolist()]
            test_dataset = dataset[test_index.tolist()]

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
            num_patients = len(test_loader)
            print(f"Number of patients in the test loader: {num_patients}")

            model = gnn().to(device)
            model.reset_parameters()

            optimizer = torch.optim.Adam(model.parameters(), lr=start_lr, weight_decay=0.0005)
            for param in model.parameters():
                param.data.clamp_(-0.01, 0.01)  # Apply weight clipping
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor,
                                                                   patience=patience, min_lr=0.0000001)

            best_val_acc = 0.0
            best_test_acc = 0.0
            best_fold_f1_score = 0.0

            early_stopping_counter = 0
            early_stopping_threshold = 40  # Number of epochs without improvement to trigger early stopping

            for epoch in range(1, max_num_epochs + 1):
                lr = scheduler.optimizer.param_groups[0]['lr']
                train(train_loader, model, optimizer, device, class_weights_tensor)
                val_acc = test(val_loader, model, device)
                scheduler.step(val_acc)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test(test_loader, model, device) * 100.0
                    best_model_state_dict = model.state_dict()
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1

                if early_stopping_counter >= early_stopping_threshold:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

            # Load the best model state dict
            model.load_state_dict(best_model_state_dict)
            # Save the best model
            torch.save(model.state_dict(), 'hemagraph.pt')

            # Evaluate on the entire test set
            model.eval()
            preds = []
            labels = []
            for data in test_loader:
                data = data.to(device)
                output = model(data)
                predss=output.max(dim=1)[1].cpu().numpy()
                labelss=data.y.cpu().numpy()
                print(len(labelss))
                idx=label0_count.index(len(labelss))+1
                precision, recall, f1, _ = precision_recall_fscore_support(labelss, predss, average='weighted', zero_division=1)

                if idx not in patient_dict.keys():
                    patient_dict[idx]=dict()
                    patient_dict[idx]['f1']=[f1]
                    patient_dict[idx]['pred']=[predss]
                    patient_dict[idx]['label']=[labelss]
                else:
                    patient_dict[idx]['f1'].append(f1)
                    patient_dict[idx]['pred'].append(predss)
                    patient_dict[idx]['label'].append(labelss)




    return patient_dict


patient_dict=gnn_evaluation(HemaGraph, max_num_epochs=1000, batch_size=1, start_lr=0.01, num_repetitions=7, all_std=True)

# Initialize a list to store the ratios for each label across all patients
average_ratio_per_label = []

# Initialize a list to store the percentages of corrected labels for each class
percentage_corrected_labels = []

for key in patient_dict.keys():
        idx = patient_dict[key]['f1'].index(max(patient_dict[key]['f1']))
        df = pd.read_csv(f"Data_original/Case_{key}.csv")
        df['predicted label'] = patient_dict[key]['pred'][idx]
        directory = "Data_predicted_gat"
        if not os.path.exists(directory):
            os.makedirs(directory)
        df.to_csv(f"{directory}//Case_{key}.csv", index=False)

        # Compute metrics for each patient
        conf_matrix = confusion_matrix(patient_dict[key]['label'][idx], patient_dict[key]['pred'][idx])
        precision = precision_score(patient_dict[key]['label'][idx], patient_dict[key]['pred'][idx], average='weighted', zero_division=1)
        recall = recall_score(patient_dict[key]['label'][idx], patient_dict[key]['pred'][idx], average='weighted', zero_division=1)
        accuracy = accuracy_score(patient_dict[key]['label'][idx], patient_dict[key]['pred'][idx])
        f1 = f1_score(patient_dict[key]['label'][idx], patient_dict[key]['pred'][idx], average='weighted', zero_division=1)

        print(f"Metrics for Patient {key}:")
        print(f"Confusion Matrix:\n{conf_matrix}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # Compute the average precision, recall, and F1 score across all patients excluding patient 30
        average_precision = np.mean([precision_score(patient_dict[key]['label'][idx], patient_dict[key]['pred'][idx], average='weighted', zero_division=1) for key in patient_dict.keys() if key != 30])
        average_recall = np.mean([recall_score(patient_dict[key]['label'][idx], patient_dict[key]['pred'][idx], average='weighted', zero_division=1) for key in patient_dict.keys() if key != 30])
        average_f1 = np.mean([f1_score(patient_dict[key]['label'][idx], patient_dict[key]['pred'][idx], average='weighted', zero_division=1) for key in patient_dict.keys() if key != 30])

        print("\nAverage Metrics Across All Patients:")
        print(f"Average Precision: {average_precision:.4f}")
        print(f"Average Recall: {average_recall:.4f}")
        print(f"Average F1 Score: {average_f1:.4f}")

        # Calculate ratio of correct predictions for each label
        total_right_cells = np.sum(np.diag(conf_matrix))
        ratio_per_label = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)

        for i, label in enumerate(range(conf_matrix.shape[0])):
            print(f"Label {label}:")
            print(f"Ratio of Correct Predictions: {ratio_per_label[i]:.4f}")

            # Add the ratio to the list for computing the average later
            if len(average_ratio_per_label) <= i:
                average_ratio_per_label.append([ratio_per_label[i]])
            else:
                average_ratio_per_label[i].append(ratio_per_label[i])

        print("-" * 50)

        # Calculate the percentage of corrected labels for each class
        percentage_corrected = np.diag(conf_matrix) / np.sum(conf_matrix, axis=0)
        percentage_corrected_labels.append(percentage_corrected)

# Calculate the average ratio for each label across all patients
average_ratio_per_label = np.mean(average_ratio_per_label, axis=1)

# Print the average ratios excluding Patient 30
print("\nAverage Ratios Across All Patients:")
for i, average_ratio in enumerate(average_ratio_per_label):
    print(f"Label {i}: {average_ratio:.4f}")

