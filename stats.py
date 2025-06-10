import sklearn.metrics as metrics
import torch
from torch.autograd import Variable
from torch_geometric.loader import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
import numpy as np
import os.path as osp
from torch_geometric.datasets import UPFD
from PHEME_Dataset import PHEME_Dataset
from WEIBO_Dataset import WEIBO_Dataset
from tqdm import tqdm

from torch_geometric.transforms import ToUndirected
import networkx as nx
import argparse


# Hyperparameters & Setup
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='gossipcop')

args = parser.parse_args()

target_dataset = args.dataset

if target_dataset == 'pheme' or target_dataset == 'PHEME':
    dataset = PHEME_Dataset()
if target_dataset == 'weibo' or target_dataset == 'WEIBO':
    dataset = WEIBO_Dataset()
if target_dataset == "politifact" or target_dataset == "gossipcop":
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'UPFD')
    train_dataset = UPFD(path, target_dataset, 'bert', 'test', ToUndirected())
    val_dataset = UPFD(path, target_dataset, 'bert', 'val', ToUndirected())
    test_dataset = UPFD(path, target_dataset, 'bert', 'train', ToUndirected())
    dataset = []
    for b in DataLoader(train_dataset, batch_size=1):
        for i in range(b.num_graphs):
            dataset.append(b[i])
    for b in DataLoader(val_dataset, batch_size=1):
        for i in range(b.num_graphs):
            dataset.append(b[i])
    for b in DataLoader(test_dataset, batch_size=1):
        for i in range(b.num_graphs):
            dataset.append(b[i])




def average_branching_factor(graph):
    # Initialize the total number of branches and total number of nodes
    total_branches = 0
    total_nodes = 0

    # Iterate over each node in the graph
    for node in graph.nodes():
        successors = graph.neighbors(node)
        num_successors = len(list(successors))
        total_branches += num_successors
        total_nodes += 1

    # Compute the average branching factor
    if total_nodes > 0:
        avg_branching_factor = total_branches / total_nodes
    else:
        avg_branching_factor = 0

    return avg_branching_factor

def compute_depth(graph, start_node = 0):
    # Create a stack to keep track of the nodes to visit
    stack = [(start_node, 0)]
    visited = set()

    # Initialize the maximum depth to 0
    max_depth = 0

    # Perform a depth-first search
    while stack:
        node, depth = stack.pop()

        # Update the maximum depth if necessary
        max_depth = max(max_depth, depth)

        # Add the node to the visited set
        visited.add(node)

        # Get the neighbors of the current node
        neighbors = graph.neighbors(node)

        # Add unvisited neighbors to the stack
        for neighbor in neighbors:
            if neighbor not in visited:
                stack.append((neighbor, depth + 1))

    return max_depth

# branching-factor list for the whole dataset.
bfs_real = []
bfs_fake = []
# depths
depths_real = []
depths_fake = []
# nodes
nodes_real = []
nodes_fake = []
# edges
edges_real = []
edges_fake = []

number_of_graphes=0

max_depth_for_real = 0
max_depth_for_fake = 0

for data in tqdm(dataset):
    # Extract the edge indices from the Data object
    edge_index = data.edge_index
    # Create a NetworkX graph from the edge indices
    graph = nx.Graph()
    graph.add_edges_from(edge_index.numpy().T)
    # Call the function to compute the average branching factor
    if data.y.item() == 0:
        number_of_graphes +=1
        
        bfs_real.append(average_branching_factor(graph))
        # depth
        depths_real.append(compute_depth(graph))

        # nodes
        nodes_real.append(graph.number_of_nodes())

        # edges
        edges_real.append(graph.number_of_edges())

        # get the max-depth to visilize the graph.
        if compute_depth(graph) > max_depth_for_real:
            max_depth_for_real = compute_depth(graph)
            #save the graph
            nx.write_gexf(graph, path="pheme_largest_real.gexf")
            #print("real",graph.number_of_nodes())

    else: 
        bfs_fake.append(average_branching_factor(graph))
        # depth
        depths_fake.append(compute_depth(graph))

        # nodes
        nodes_fake.append(graph.number_of_nodes())

        # edges
        edges_fake.append(graph.number_of_edges())
        # save the graph
         # get the max-depth to visilize the graph.
        if compute_depth(graph) > max_depth_for_fake:
            max_depth_for_fake = compute_depth(graph)
            #save the graph
            nx.write_gexf(graph, path="pheme_largest_fake.gexf")
            #print("fake",graph.number_of_nodes())
    
print("largest depth for real", max_depth_for_real)
print("largest depth for fake", max_depth_for_fake)


print('number of real',number_of_graphes,' number of fakes', len(dataset)-number_of_graphes)

print('all Dataset')
print("branching-factor",target_dataset, np.mean(bfs_real + bfs_fake))
print("branching-factor max",target_dataset, np.max(bfs_real + bfs_fake))
print("branching-factor min",target_dataset, np.min(bfs_real + bfs_fake))
print("branching-factor std",target_dataset, np.std(bfs_real + bfs_fake))
print("depth :", target_dataset, np.mean(depths_real+depths_fake))
print("depth max:", target_dataset, np.max(depths_real+depths_fake))
print("depth min:", target_dataset, np.min(depths_real+depths_fake))
print("depth std:", target_dataset, np.std(depths_real+depths_fake))
print("nodes :", target_dataset, sum(nodes_real+nodes_fake))
print("edges :", target_dataset, sum(edges_real+edges_fake))

separator = '-' * 79
print(separator)
print('Real Dataset:')
print("branching-factor",target_dataset, np.mean(bfs_real ))
print("branching-factor max",target_dataset, np.max(bfs_real ))
print("branching-factor min",target_dataset, np.min(bfs_real ))
print("branching-factor std",target_dataset, np.std(bfs_real ))
print("depth :", target_dataset, np.mean(depths_real))
print("depth max:", target_dataset, np.max(depths_real))
print("depth min:", target_dataset, np.min(depths_real))
print("depth std:", target_dataset, np.std(depths_real))
print("nodes :", target_dataset, sum(nodes_real))
print("edges :", target_dataset, sum(edges_real))

print(separator)

print("branching-factor",target_dataset, np.mean(bfs_fake))
print("branching-factor max",target_dataset, np.max(bfs_fake))
print("branching-factor min",target_dataset, np.min(bfs_fake))
print("branching-factor std",target_dataset, np.std(bfs_fake))
print("depth :", target_dataset, np.mean(depths_fake))
print("depth max:", target_dataset, np.max(depths_fake))
print("depth min:", target_dataset, np.min(depths_fake))
print("depth std:", target_dataset, np.std(depths_fake))
print("nodes :", target_dataset, sum(nodes_fake))
print("edges :", target_dataset, sum(edges_fake))

