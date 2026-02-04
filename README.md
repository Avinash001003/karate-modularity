# Spectral Modularity Optimization: Recursive Bisection on the Karate Club Graph

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Academic-Project-orange)

## Project Overview
This project implements a **Recursive Spectral Bisection** algorithm to detect community structures within complex networks. Using **Zachary's Karate Club** dataset as a benchmark, the algorithm maximizes the **Modularity Score ($Q$)** to mathematically partition the graph into subgroups without prior knowledge of the ground truth labels.

This implementation focuses on the intersection of **Linear Algebra** and **Network Theory**, specifically leveraging the spectral properties of the Modularity Matrix.

## Mathematical Foundation
The core of this project relies on the **Modularity Matrix ($B$)**, defined as:

$$B_{ij} = A_{ij} - \frac{k_i k_j}{2m}$$

Where:
- $A_{ij}$ is the Adjacency Matrix.
- $k_i, k_j$ are the degrees of nodes $i$ and $j$.
- $m$ is the total number of edges.

The algorithm utilizes the **Leading Eigenvector** of this matrix to perform optimal graph partitioning, analogous to energy minimization states in physical systems.

### Algorithm Methodology
1.  **Eigen-Decomposition:** Compute the leading eigenvector of $B$ for the active subgraph.
2.  **Spectral Partitioning:** Split nodes based on the sign of the eigenvector components ($\vec{u}_i > 0$ vs $\vec{u}_i < 0$).
3.  **Recursive Refinement:** Repeat the process for each community until the leading eigenvalue $\lambda \le 0$ (no further modularity gain).

## Repository Contents
- `main.py`: Core recursive algorithm and visualization logic.
- `analysis.ipynb`: Interactive Jupyter Notebook for step-by-step analysis.
- `outputs/`: Generated community visualizations and JSON partitions.
- `requirements.txt`: Python dependencies.

## Key Results
The algorithm successfully reconstructs the historical social fracture of the club ("Mr. Hi" vs. "Officer" factions) purely through topological analysis, demonstrating the effectiveness of spectral methods in detecting latent community structures.

## 🎓 Academic Context
This project was developed as part of the **Graph Theory & Network Analysis** curriculum at **IISER Thiruvananthapuram**.

**Supervisor:** Dr. Saptarishi Bej  
**Course:** DSC212 (Graph Theory Module)

---
**Author:** Avinash Kumar Thakur  
*BS-MS Dual Degree, IISER Thiruvananthapuram*