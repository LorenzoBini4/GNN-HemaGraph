import torch

# Load the list of graphs
dataset = torch.load('data/A_graph.pt')  

# Print the number of graphs in the final list
print(f"Number of initial graphs in the list: {len(dataset)}")

# Define a function to compute class distribution percentages
def compute_class_distribution_percentage(mask, labels):
    class_count = torch.zeros(labels.max().item() + 1)
    total_nodes = 0
    for i, label in enumerate(labels):
        if mask[i]:
            class_count[label.item()] += 1
            total_nodes += 1
    return (class_count / total_nodes) * 100  # Calculate percentage

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

    # Append class weights to the list
    class_weights_tensor.append(class_weights)

    # Print class weights for the current graph (optional)
    print(f"Graph {i + 1} Class Weights:", class_weights)
    print("-" * 50)

# Now, class_weights_tensor contains 30 different class_weights tensors
#print(class_weights_tensor)
