# Spectral Modularity Recursive Bisection on the Karate Club Graph
**DSC212: Graph Theory Module — Research Assignment**  
Author: Avinash Kumar Thakur  
Repository: `karate-modularity`  
Date: 2025-10-27

---

## Abstract
This repository contains an implementation of the spectral modularity method for community detection, applied to Zachary’s Karate Club network. The goal is to reconstruct meaningful communities using the modularity objective and a recursive spectral bisection procedure. The implementation is written in Python and delivered as a runnable Jupyter notebook (`Karate_modularity_notebook.ipynb`) that executes top-to-bottom without manual edits.

---

## Contents
- `Karate_modularity_notebook.ipynb` — Complete Jupyter notebook: code, visualizations, metric evolution plots, and discussion.  
- `Karate_modularity_notebook.py` — Script version (same code in a .py file; can be converted to .ipynb using jupytext).  
- `outputs/` — Generated figures and `final_communities.json` produced by running the notebook.  
- `requirements.txt` — Required Python packages and versions.

---

## Requirements
This project was developed and tested with:
- Python 3.9+ (3.10 recommended)
- `networkx`
- `numpy`
- `matplotlib`
- `jupyter` (for running the notebook)
- `jupytext` (if converting between .py and .ipynb)

Install requirements:

```bash
python -m pip install -r requirements.txt
