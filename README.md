# Glacier: Decoding the causal drivers of spatial cellular topology
![Figure 1](fig1.png)

Glacier is a python library to leverage spatial Granger causality to infer transcriptional and signaling relationships based on tissue organization. By combining GASTON’s global isodepth coordinate with Velorama’s graph-based causal inference framework, we enable bidirectional inference of regulatory relationships along spatial axes, identifying transcription factor target interactions and ligand-receptor pairs that operate across spatial domains.

**Our program is designed specifically for single-cell, spatial transcriptomics data.** Glacier constructs a directed acyclic graph (DAG) based on global spatial coordinates, by first using [GASTON](https://github.com/raphael-group/GASTON) to compute isodepth for each cell. Then, we test for Granger causality by using spatial-data adapted [Velorama](https://github.com/rs239/velorama). We are also able to invert each cell's isodepth, allowing information to flow in the other direction. 

## Installation:

We recommend first setting up a conda environment with ```python>=3.10```. To run Glacier, it is important to first install both GASTON and Velorama, with 
```bash
pip install pip install gaston-spatial
``` 
and 
```bash
git clone https://github.com/rs239/velorama.git
cd ./velorama
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
pip install .
``` 
respectively.  

## Tut

