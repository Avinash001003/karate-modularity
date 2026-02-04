"""
Karate Club Modularity: Recursive Spectral Bisection
----------------------------------------------------
This script implements a physics-inspired community detection algorithm 
using the 'Modularity Matrix' (B). It recursively splits the graph 
by analyzing the leading eigenvector of B, maximizing the modularity 
score (Q) at each step.

Methodology:
1. Compute the Modularity Matrix B = A - (k_i * k_j) / 2m
2. Compute the leading eigenvector of B for the current subgraph.
3. Split nodes based on the sign of the eigenvector components (+/-).
4. Repeat until no positive modularity gain is possible.
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import json
import itertools

# --- Configuration & Parameters ---
SEED = 42
OUTDIR = 'outputs'
os.makedirs(OUTDIR, exist_ok=True)

# --- Data Loading ---
print("Loading Zachary's Karate Club Graph...")
G = nx.karate_club_graph()
n = G.number_of_nodes()
m = G.number_of_edges()

# Adjacency Matrix (A) and Degrees (k)
A = nx.to_numpy_array(G)
degrees = A.sum(axis=1)

# --- Precompute Global Modularity Matrix ---
# B_ij = A_ij - (k_i * k_j) / (2m)
# This represents the difference between the actual structure (A) 
# and the expected structure of a random graph with the same degree distribution.
B_global = A - np.outer(degrees, degrees) / (2.0 * m)

# Fixed layout for consistent plotting across iterations
pos = nx.spring_layout(G, seed=SEED)


# --- Linear Algebra Helpers ---
def leading_eigenpair(M):
    """
    Computes the largest algebraic eigenvalue and its corresponding eigenvector.
    """
    # 'eigh' is optimized for symmetric matrices and returns eigenvalues in ascending order.
    vals, vecs = np.linalg.eigh(M)
    # The last element is the largest eigenvalue (leading)
    return vals[-1], vecs[:, -1]


# --- Core Algorithm: Spectral Split ---
def split_on_nodes(node_list, B_global):
    """
    Attempts to split a subgraph (defined by node_list) using the 
    restricted modularity matrix.
    
    Returns:
        can_split (bool): True if a valid split maximizes modularity.
        Cplus (list): Nodes with positive eigenvector components.
        Cminus (list): Nodes with negative eigenvector components.
        lam (float): The leading eigenvalue (modularity gain).
    """
    if len(node_list) <= 1:
        return False, None, None, 0.0

    # Extract the sub-matrix of B corresponding only to these nodes
    ix_grid = np.ix_(node_list, node_list)
    M = B_global[ix_grid]

    # Compute the leading eigenpair of this restricted matrix
    lam, u = leading_eigenpair(M)

    # Physics Check: If eigenvalue <= 0, splitting reduces or doesn't change modularity.
    # We use a small epsilon (1e-5) to handle floating point noise.
    if lam <= 1e-5:
        return False, None, None, lam

    # Partition based on the sign of the eigenvector components
    Cplus = [node_list[i] for i, val in enumerate(u) if val > 0]
    Cminus = [node_list[i] for i, val in enumerate(u) if val <= 0]

    # Safety Check: Ensure we don't return an empty community
    if len(Cplus) == 0 or len(Cminus) == 0:
        return False, None, None, lam

    return True, Cplus, Cminus, lam


# --- Metric Analysis ---
def compute_metrics_for_subgraph(G_sub):
    """
    Calculates centrality metrics for a given subgraph.
    Note: 'Betweenness' will naturally decrease as graphs get smaller/fragmented.
    """
    return {
        'degree': nx.degree_centrality(G_sub),
        'betweenness': nx.betweenness_centrality(G_sub),
        'closeness': nx.closeness_centrality(G_sub),
        'clustering': nx.clustering(G_sub),
    }


# --- Visualization ---
# Use 'tab20' colormap to support up to 20 distinct communities clearly
color_cycle = plt.cm.tab20.colors

def draw_communities(G, communities, iteration, pos=pos, save=True):
    plt.figure(figsize=(8, 6))
    
    # Create a map of Node ID -> Community ID
    node_to_comm = {}
    for cid, group in enumerate(communities):
        for node in group:
            node_to_comm[node] = cid
            
    # Assign colors
    node_colors = []
    for node in G.nodes():
        cid = node_to_comm.get(node, 0)
        # Use modulo to cycle colors if communities > 20
        node_colors.append(color_cycle[cid % len(color_cycle)])
        
    # Draw Graph
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=350, edgecolors='k')
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=9, font_color='white', font_weight='bold')
    
    plt.title(f'Iteration {iteration}: {len(communities)} Communities Found')
    plt.axis('off')
    
    if save:
        filename = os.path.join(OUTDIR, f'communities_iter_{iteration:02d}.png')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  > Saved figure: {filename}")
    
    plt.close()


# --- Main Recursive Logic ---
def recursive_bisect(B_global, G, pos=None):
    """
    Recursively partitions the graph.
    Returns:
        final_communities (list of lists): The leaf nodes of the partition tree.
        metrics_by_iter (dict): Nested dictionary of metrics tracked over time.
    """
    initial_nodes = list(G.nodes())
    queue = [initial_nodes]
    final_communities = []

    # Storage: iteration -> metric name -> node -> value
    metrics_by_iter = defaultdict(lambda: defaultdict(dict))
    iteration = 0

    print("\n--- Starting Recursive Bisection ---")

    while queue:
        # 1. Pop the next candidate community
        C = queue.pop(0)
        
        # 2. Visualize the current state (Processed + Queue + Current)
        current_state = final_communities + queue + [C]
        draw_communities(G, current_state, iteration, pos=pos)
        
        # 3. Attempt to split
        can_split, Cplus, Cminus, lam = split_on_nodes(C, B_global)
        
        # 4. Compute Metrics (Physical Analysis of the Substructure)
        subG = G.subgraph(C).copy()
        metrics = compute_metrics_for_subgraph(subG)
        
        # Store metrics (keyed by global node ID)
        for metric_name, values_dict in metrics.items():
            for node_id, val in values_dict.items():
                metrics_by_iter[iteration][metric_name][node_id] = val

        # 5. Decision Tree
        if not can_split:
            # Physics: Energy (Modularity) cannot be minimized further.
            print(f"Iter {iteration}: Finalized community of size {len(C)} (Eigenval: {lam:.4f})")
            final_communities.append(C)
        else:
            print(f"Iter {iteration}: Split community of size {len(C)} -> {len(Cplus)} & {len(Cminus)} (Eigenval: {lam:.4f})")
            queue.append(Cplus)
            queue.append(Cminus)
        
        iteration += 1

    return final_communities, metrics_by_iter


# --- Execution Block ---
if __name__ == "__main__":
    final_comms, metrics_by_iter = recursive_bisect(B_global, G, pos=pos)
    
    print('\nFinal Partitioning:', final_comms)

    # --- Plot Metric Evolution ---
    print("\nGenerating Metric Evolution Plots...")
    iterations = sorted(metrics_by_iter.keys())
    all_nodes = list(G.nodes())

    for metric in ['degree', 'betweenness', 'closeness', 'clustering']:
        plt.figure(figsize=(10, 6))
        
        # Only plot nodes that were active in the calculation at some point
        for node in all_nodes:
            y_vals = []
            x_vals = []
            for it in iterations:
                val = metrics_by_iter[it][metric].get(node, None)
                if val is not None:
                    y_vals.append(val)
                    x_vals.append(it)
            
            if y_vals:
                plt.plot(x_vals, y_vals, marker='o', markersize=3, alpha=0.5, label=f'Node {node}')

        plt.xlabel('Iteration Step')
        plt.ylabel(f'{metric.capitalize()} (Local)')
        plt.title(f'Evolution of {metric.capitalize()} Centrality During Partitioning')
        
        # Only show legend if the graph isn't too cluttered
        if len(all_nodes) <= 10:
            plt.legend()
            
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f'metric_evolution_{metric}.png'), dpi=150)
        plt.close()

    # --- Save Results to JSON ---
    # Important: Convert numpy ints to python ints for JSON serialization
    clean_comms = [[int(n) for n in comm] for comm in final_comms]
    
    json_path = os.path.join(OUTDIR, 'final_communities.json')
    with open(json_path, 'w') as f:
        json.dump(clean_comms, f, indent=4)
        
    print(f"Analysis Complete. Data saved to {json_path}")