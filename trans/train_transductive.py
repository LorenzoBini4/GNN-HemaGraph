import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from torch_geometric.data import DataLoader
from utils.mask import class_weights_tensor
from torch_geometric.explain import Explainer, Explanation
from torch_geometric.explain import GNNExplainer, DummyExplainer, CaptumExplainer, PGExplainer, AttentionExplainer
from torch_geometric.explain.config import ExplainerConfig, ModelMode
from sklearn.metrics import precision_recall_fscore_support
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import MessagePassing
from torch.nn.functional import softmax
import graphviz
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set seeds for reproducibility
seed_value = 77

# Set seed for random module
random.seed(seed_value)

# Set seed for NumPy
np.random.seed(seed_value)

# Set seed for PyTorch
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

# Load the list of masked graphs
masked_graphs = torch.load('masked_graph.pt')

# Define the model
class HemaGraph(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels=16, num_heads=8):
        super(HemaGraph, self).__init__()
        self.conv1 = GATConv(num_features, hidden_channels, heads=num_heads)
        self.conv2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads)
        self.conv3 = GATConv(hidden_channels * num_heads, num_classes, heads=1)
        self.dropout = torch.nn.Dropout(0.4)
        self.relu = torch.nn.ReLU()
        #self.leaky_relu = torch.nn.LeakyReLU(0.4)
        

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HemaGraph(num_features=12, num_classes=5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=0.0000001)

# Add a new random mask to each graph  
for graph in masked_graphs:
  n = len(graph.y)
  mask = [random.random() < 0.5 for _ in range(n)]
  graph.rand_mask = torch.tensor(mask, dtype=torch.bool)

best_val_loss = float('inf')
patience = 10 
counter = 0
class_weights_tensor = [weights.to(device) for weights in class_weights_tensor]

'''
The code belove masking 50% of the nodes inside the train_mask. 
It combines the train_mask with the rand_mask using the bitwise AND operator (&) 
to create the full_mask. The full_mask will only have True values for the nodes 
that are both in the train_mask and the rand_mask, effectively masking 50% of the nodes in the train_mask.
Same for val_mask and test_mask.
'''
for epoch in range(1, 101):

    model.train()
    total_loss = 0
    for graph in masked_graphs:

        # Combine existing mask with new random mask
        full_mask = graph.train_mask & graph.rand_mask

        data = graph.to(device)
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)
        current_weights = class_weights_tensor[masked_graphs.index(graph)]
        criterion = nn.NLLLoss(weight=current_weights)
        loss = criterion(out[full_mask], data.y[full_mask])

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    average_train_loss = total_loss / len(masked_graphs)

    # Validation
    model.eval()
    val_loss = 0

    for graph in masked_graphs:

        # Combine existing mask with new random mask
        full_mask = graph.val_mask & graph.rand_mask

        data = graph.to(device)
        with torch.no_grad():
            out = model(data.x, data.edge_index)

        current_weights = class_weights_tensor[masked_graphs.index(graph)]
        criterion = nn.NLLLoss(weight=current_weights)
        loss = criterion(out[full_mask], data.y[full_mask])
        val_loss += loss.item()

    average_val_loss = val_loss / len(masked_graphs)
    print(f'Epoch: {epoch}, Training Loss: {average_train_loss:.4f}, Validation Loss: {average_val_loss:.4f}')

    # Update the learning rate based on validation loss
    scheduler.step(average_val_loss)
    if average_val_loss < best_val_loss:
        best_val_loss = average_val_loss
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f'Early stopping after {epoch} epochs.')
            break

print('Training completed.')

accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

for graph in masked_graphs:
    full_mask = graph.test_mask & graph.rand_mask
    data = graph.to(device)
    with torch.no_grad():
        out = model(data.x, data.edge_index)
    
    predicted_labels = torch.argmax(out[full_mask], dim=1)
    true_labels = data.y[full_mask]
    
    accuracy = accuracy_score(true_labels.cpu(), predicted_labels.cpu())
    precision = precision_score(true_labels.cpu(), predicted_labels.cpu(), average='weighted', zero_division=1)
    recall = recall_score(true_labels.cpu(), predicted_labels.cpu(), average='weighted', zero_division=1)
    f1 = f1_score(true_labels.cpu(), predicted_labels.cpu(), average='weighted', zero_division=1)
    
    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)

for i in range(len(masked_graphs)):
    print(f'Graph {i+1}:')
    print(f'Accuracy: {accuracy_scores[i]:.4f}')
    print(f'Precision: {precision_scores[i]:.4f}')
    print(f'Recall: {recall_scores[i]:.4f}')
    print(f'F1 Score: {f1_scores[i]:.4f}')
    print()

print('Training, validation, and testing completed.')
average_accuracy = sum(accuracy_scores) / len(accuracy_scores)
average_precision = sum(precision_scores) / len(precision_scores)
average_recall = sum(recall_scores) / len(recall_scores)
average_f1 = sum(f1_scores) / len(f1_scores)

print(f'Average Accuracy: {average_accuracy:.4f}')
print(f'Average Precision: {average_precision:.4f}')
print(f'Average Recall: {average_recall:.4f}')
print(f'Average F1 Score: {average_f1:.4f}')

######### EMBEDDINGS WITH t-SNE #########

model.eval()
data22 = masked_graphs[22]

import matplotlib.pyplot as plt
import umap
import numpy as np
from sklearn.manifold import TSNE
from matplotlib.lines import Line2D

label_to_color_map = {0: "red", 1: "blue", 2: "green", 3: "black", 4: "yellow"}
t_sne_embeddings = TSNE(n_components=2, perplexity=220, method='barnes_hut').fit_transform(data22.x.detach().cpu().numpy())
num_classes = 5
fig = plt.figure(figsize=(12, 8), dpi=100)
for class_id in range(num_classes):
    plt.scatter(t_sne_embeddings[data22.y.cpu() == class_id, 0], t_sne_embeddings[data22.y.cpu() == class_id, 1], s=4, color=label_to_color_map[class_id], edgecolors='black', linewidths=0.2)

legend_labels = ['T Lymphocytes', 'B Lymphocytes', 'Monocytes', 'Mast Cells', 'Hematopoietic']
legend_elements = [Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=label_to_color_map[class_id], markersize=5) for class_id, label in enumerate(legend_labels)]
plt.legend(handles=legend_elements)

plt.savefig('TRANS23.png', dpi=100)
plt.show()

######################################## ATTENTION EXPLAINER ########################################
# AttentionExplainer uses attention coefficients to determine edge weights/opacities.
# This is why we used GATConvs in our model. We could have also used GATv2Conv
# or TransformerConv.
attention_explainer = Explainer(
    model=model,
    # AttentionExplainer takes an optional reduce parameter. The reduce parameter
    # allows you to set how you want to aggregate attention coefficients over layers
    # and heads. The explainer will then aggregate these values using this
    # given method to determine the edge_mask (we use the default 'max' here).
    algorithm=AttentionExplainer(),
    explanation_type='model',
    # Like PGExplainer, AttentionExplainer also does not support node_mask_type
    edge_mask_type='object',
    model_config=dict(mode='multiclass_classification', task_level='node', return_type='log_probs'),
)

data = masked_graphs[22]
node_index=10
attention_explanation = attention_explainer(data.x, data.edge_index, index=node_index)
attention_explanation.visualize_graph("attention_graph_10.png", backend="graphviz")
plt.imshow(plt.imread("attention_graph_10.png"))


################################### EXPLAINABILITY ###################################
explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=200),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(
        mode='multiclass_classification',
        task_level='node',
        return_type='log_probs',
    ),
)

data = masked_graphs[22]
node_index = 10
explanation = explainer(data.x, data.edge_index, index=node_index)

print(f'Generated explanations in {explanation.available_explanations}')

path = 'feature_importance.png'
explanation.visualize_feature_importance(path, top_k=10)
print(f"Feature importance plot has been saved to '{path}'")

path = 'subgraph.pdf'
explanation.visualize_graph(path)
print(f"Subgraph visualization plot has been saved to '{path}'")


################################### DEGREES VISUALIZATION ##########################

from torch_geometric.utils import degree
import numpy as np
import matplotlib.pyplot as plt

# Get model's classifications
out = model(data.x, data.edge_index)

# Move the 'out' tensor from GPU to CPU
out = out.cpu()

# Move the 'data.y' tensor from GPU to CPU
data.y = data.y.cpu()

# Calculate the degree of each node
degrees = degree(data.edge_index[0]).cpu().numpy()

# Store accuracy scores and sample sizes
accuracies = []
sizes = []

# Accuracy for degrees between 0 and 5
for i in range(0, 6):
    mask = np.where(degrees == i)[0]
    accuracies.append(accuracy_score(out.argmax(dim=1)[mask], data.y[mask]))
    sizes.append(len(mask))

# Accuracy for degrees > 5
mask = np.where(degrees > 5)[0]
accuracies.append(accuracy_score(out.argmax(dim=1)[mask], data.y[mask]))
sizes.append(len(mask))

# Bar plot
fig, ax = plt.subplots(figsize=(18, 9))
ax.set_xlabel('Node degree')
ax.set_ylabel('Accuracy score')
ax.set_facecolor('#EFEEEA')
plt.bar(['0','1','2','3','4','5','>5'],
        accuracies,
        color='#0A047A')
for i in range(0, 7):
    plt.text(i, accuracies[i], f'{accuracies[i]*100:.2f}%',
             ha='center', color='#0A047A')
for i in range(0, 7):
    plt.text(i, accuracies[i]//2, sizes[i],
             ha='center', color='white')

# Save the plot as "degree_accuracy.png" with dpi=100
plt.savefig('degree_accuracy.png', dpi=100)

#########################
