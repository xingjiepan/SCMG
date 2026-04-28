[![PyPI](https://img.shields.io/pypi/v/scmg.svg)](https://pypi.org/project/scmg)
[![Docs](https://readthedocs.org/projects/SCMG/badge/?version=latest)](https://scmg.readthedocs.io/en/latest/index.html)

# Single Cell Manifold Generator (SCMG)

**SCMG** is a suite of deep learning models designed to interpret, generate, and predict the molecular basis of cell states and their transitions.

## Key Features

- **Global Manifold Construction**  
  Build a well-integrated reference manifold of single-cell transcriptional states that captures cell-state relationships and gene expression patterns. The global gene expression patterns can be visualized [here](https://xingjiepan.github.io/SCMG/).

- **Zero-Shot Dataset Integration**  
  Integrate new scRNA-seq datasets without the need for model retraining.

- **Zero-Shot Cell Projection**  
  Project single-cells onto the global manifold for downstream analysis and comparison.

- **Cell State Trajectory Generation**  
  Generate continuous trajectories to model transitions between cell states.

- **Causal Gene Prediction**  
  Identify candidate causal genes driving transitions between specific cell states.
  
- **Universal Decomposition of Perturbation Effects**  
  Decompose perturbation effects into universal principal axes of cell state transition and perturbation classes. 

- **Few-shot Prediction of Perturbation Effects**  
  Predict perturbation-induced cell state transition by few-shot learning.

## Installation and Tutorials
Full documentation is available at: https://scmg.readthedocs.io/

The scripts to reproduce the results reported in the manuscript are available [here](https://github.com/xingjiepan/SCMG_scripts).
