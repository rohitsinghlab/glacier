# Glacier: Decoding the causal drivers of spatial cellular topology
![Figure 1](fig1.png)

Glacier is a python library to leverage spatial Granger causality to infer transcriptional and signaling relationships based on tissue organization. By combining GASTON’s global isodepth coordinate with Velorama’s graph-based causal inference framework, we enable bidirectional inference of regulatory relationships along spatial axes, identifying transcription factor target interactions and ligand-receptor pairs that operate across spatial domains.

**Our program is designed specifically for single-cell, spatial transcriptomics data.** Glacier constructs a directed acyclic graph (DAG) based on global spatial coordinates, by first using [GASTON](https://github.com/raphael-group/GASTON) to compute isodepth for each cell. Then, we test for Granger causality by using spatial-data adapted [Velorama](https://github.com/rs239/velorama). We are also able to invert each cell's isodepth, allowing information to flow in the other direction. 

## Installation:

We recommend first setting up a conda environment with ```python>=3.10```. To run Glacier, it is important to first install both GASTON and Velorama, with 
```bash
pip install gaston-spatial
``` 
and 
```bash
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
pip install git+https://github.com/rs239/velorama.git
``` 
respectively. Then, clone this repository with 
```bash
git clone git@github.com:rohitsinghlab/glacier.git
cd glacier
```
## Tutorial

After everything is installed, navigate to the tutorial folder. ```cerebellum_data``` is there as sample data. First download ```cerebellum_counts_mat.npy``` from [here](https://drive.google.com/drive/u/0/folders/1TaTSYs2z-Vb7-X8yLpZeokjN_52x6F1k) and put it inside the ```cerebellum_data``` folder. Then follow the tutorial, including the ```CREATE DATASETS``` portion.

With created datasets, go to ```velorama``` folder and run commands such as ```python run_modified.py -ds $dataset -dyn dag_precomputed -dev $device``` to infer transcriptional and signaling relationships. **```-dyn``` needs to remain ```dag_precomputed```**, since we use the spatial dag. ```$dataset``` is the data saved from the previous step. ```$device``` can be cpu or gpu. Other hyperparameters that can be change include maximum number of lags ```$L```, dimension of hidden layers ```$hidden```, among others.


