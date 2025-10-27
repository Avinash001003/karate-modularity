# Karate Club modularity spectral recursive bisection
# This script is intended to be used as a Jupyter notebook (.ipynb) cell-by-cell.
# It implements recursive spectral modularity partitioning, visualizations,
# metric tracking, and saves figures ready for inclusion in a report.

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os

# --- Parameters ---
SEED = 42
OUTDIR = 'outputs'
os.makedirs(OUTDIR, exist_ok=True)

# --- Load graph ---
G = nx.karate_club_graph()
n = G.number_of_nodes()
m = G.number_of_edges()
A = nx.to_numpy_array(G)
degrees = A.sum(axis=1)

# Precompute global modularity matrix B
B_global = A - np.outer(degrees, degrees) / (2.0 * m)

# Fixed layout for consistent plotting
pos = nx.spring_layout(G, seed=SEED)

# --- Linear algebra helpers ---
def leading_eigenpair(M):
    # For symmetric M, eigh is stable and returns ascending eigenvalues
    vals, vecs = np.linalg.eigh(M)
    idx = np.argmax(vals)
    return vals[idx], vecs[:, idx]

# --- Split function on a set of node indices ---
def split_on_nodes(node_list, B_global):
    """Try to split the induced subgraph on node_list using restricted modularity matrix.
    Returns: can_split (bool), (Cplus, Cminus) lists of node indices, leading eigenvalue
    """
    if len(node_list) <= 1:
        return False, None, None, None
    # build restricted modularity matrix
    M = B_global[np.ix_(node_list, node_list)]
    lam, u = leading_eigenpair(M)
    if lam <= 0:
        return False, None, None, lam
    # threshold by sign
    Cplus = [node_list[i] for i, val in enumerate(u) if val > 0]
    Cminus = [node_list[i] for i, val in enumerate(u) if val <= 0]
    # avoid empty group
    if len(Cplus) == 0 or len(Cminus) == 0:
        return False, None, None, lam
    return True, Cplus, Cminus, lam

# --- Metric collection helpers ---
def compute_metrics_for_subgraph(G_full):
    return {
        'degree': nx.degree_centrality(G_full),
        'betweenness': nx.betweenness_centrality(G_full),
        'closeness': nx.closeness_centrality(G_full),
        'clustering': nx.clustering(G_full),
    }

# --- Visualization helpers ---
import itertools
palette = list(plt.rcParams['axes.prop_cycle'].by_key()['color'])
# if more communities than palette, colors will repeat

def draw_communities(G, communities, iteration, pos=pos, save=True):
    plt.figure(figsize=(7,6))
    # map each node to a community id
    node_to_comm = {}
    for cid, C in enumerate(communities):
        for v in C:
            node_to_comm[v] = cid
    colors = [palette[node_to_comm[i] % len(palette)] for i in range(G.number_of_nodes())]
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=300)
    nx.draw_networkx_edges(G, pos, alpha=0.6)
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.title(f'Communities after iteration {iteration} (count={len(communities)})')
    plt.axis('off')
    if save:
        plt.savefig(os.path.join(OUTDIR, f'communities_iter_{iteration}.png'), dpi=150)
    plt.show()

# --- Recursive bisection with metric tracking ---

def recursive_bisect(B_global, G, pos=None):
    # start with all nodes as a single candidate community
    initial = list(range(G.number_of_nodes()))
    queue = [initial]
    final_communities = []

    # metrics storage: iteration -> metric name -> node -> value
    metrics_by_iter = defaultdict(lambda: defaultdict(dict))
    iteration = 0

    while queue:
        C = queue.pop(0)
        can_split, Cplus, Cminus, lam = split_on_nodes(C, B_global)
        # compute metrics for the current induced subgraph
        subG = G.subgraph(C).copy()
        metrics = compute_metrics_for_subgraph(subG)
        # store metrics (normalize keys to full node ids)
        for metric_name, dct in metrics.items():
            for node_in_sub, val in dct.items():
                metrics_by_iter[iteration][metric_name][node_in_sub] = val
        # Build list of current communities for visualization: take final_communities + queue + current C if not split
        current_view = final_communities + queue + [C]
        draw_communities(G, current_view, iteration, pos=pos)

        if not can_split:
            final_communities.append(C)
        else:
            # push children to queue
            queue.append(Cplus)
            queue.append(Cminus)
        iteration += 1

    return final_communities, metrics_by_iter

final_comms, metrics_by_iter = recursive_bisect(B_global, G, pos=pos)
print('Final communities:', final_comms)

# --- Plot evolution of metrics across iterations ---
# Prepare a consistent list of iterations and nodes
iterations = sorted(metrics_by_iter.keys())
all_nodes = list(range(G.number_of_nodes()))

# For each metric, build matrix: rows=iterations, cols=nodes
for metric in ['degree', 'betweenness', 'closeness', 'clustering']:
    plt.figure(figsize=(9,6))
    for node in all_nodes:
        y = [metrics_by_iter[it][metric].get(node, np.nan) for it in iterations]
        plt.plot(iterations, y, label=str(node))
    plt.xlabel('Iteration')
    plt.ylabel(metric)
    plt.title(f'Evolution of {metric} across iterations')
    # show legend only if small graph (34 nodes is okay but can be busy) - place outside
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=1)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f'metric_evolution_{metric}.png'), dpi=150)
    plt.show()

# --- Short discussion scaffold ---
print('\n--- Discussion (short) ---')
print('Review the saved community figures in the outputs/ folder and the metric evolution plots.')
print('Look for nodes whose centrality values remain high across iterations (likely hubs).')
print('Discuss how splitting isolates or groups them and whether modularity splits align with the historical Mr. Hi vs President split.')

# --- Save communities to file for reference ---
import json
with open(os.path.join(OUTDIR, 'final_communities.json'), 'w') as f:
    json.dump(final_comms, f)

# --- END ---
print('Outputs saved in the outputs/ directory.')
