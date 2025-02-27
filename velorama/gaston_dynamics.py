import sys
import os
from collections import defaultdict
import pandas as pd
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
from glmpca import glmpca

from itertools import combinations
import torch

from importlib import reload

import gaston

import seaborn as sns
import math

from gaston import neural_net,cluster_plotting, dp_related, segmented_fit
from gaston import binning_and_plotting, isodepth_scaling, run_slurm_scripts
from gaston import spatial_gene_classification, plot_cell_types, filter_genes, process_NN_output

import networkx as nx
from sklearn.neighbors import NearestNeighbors
import time

from matplotlib.patches import FancyArrowPatch


def glm_pca(num_dims, penalty, counts_mat, coords_mat):
    num_dims=8 # 2 * number of clusters
    penalty=10 # may need to increase if this is too small
    glmpca_res=glmpca.glmpca(counts_mat.T, num_dims, fam="poi", penalty=penalty, verbose=True)
    A = glmpca_res['factors'] # should be of size N x num_dims, where each column is a PC
    S=coords_mat
    S_torch, A_torch = neural_net.load_rescale_input_data(S,A)
    
    return S_torch, A_torch

def gaston_model(out_dir, S_torch, A_torch, isodepth_arch=[20,20], expression_fn_arch=[20,20], num_epochs=1000, checkpoint=500, 
                 optimizer='adam', num_restarts=30):
    seed_list=range(num_restarts)
    for seed in seed_list:
        print(f'training neural network for seed {seed}')
        out_dir_seed=f"{out_dir}/rep{seed}"
        os.makedirs(out_dir_seed, exist_ok=True)
        mod, loss_list = neural_net.train(S_torch, A_torch,
                              S_hidden_list=isodepth_arch, A_hidden_list=expression_fn_arch, 
                              epochs=num_epochs, checkpoint=checkpoint, 
                              save_dir=out_dir_seed, optim=optimizer, seed=seed, save_final=True)
    gaston_model, A, S= process_NN_output.process_files(out_dir)
    
    return gaston_model, A, S


def create_cell_type_df(input_data):
    cell_type_df=pd.read_csv(input_data, index_col=0)
    cell_type_mod_df = cell_type_df.copy()
    cell_type_mod_df['Cell_Type'] = cell_type_mod_df.idxmax(axis=1)
    cell_type_mod_df = cell_type_mod_df[['Cell_Type']]
    
    return cell_type_mod_df

def calculate_isodepth(gaston_model, A, S, coords_mat = None, num_layers=4,
    scale_factor=64/100, flip=False, adjust = True):
    
#    mod = 1
#    if flip:
#        mod = -1
        
    gaston_isodepth, gaston_labels = dp_related.get_isodepth_labels(gaston_model, A, S, num_layers)
    if adjust:
        gaston_isodepth = isodepth_scaling.adjust_isodepth(gaston_isodepth, gaston_labels, coords_mat, q_vals=[0.2, 0.05, 0.15, 0.3], scale_factor=scale_factor)
        
    if flip:
        
        gaston_isodepth = np.max(gaston_isodepth) - gaston_isodepth
        # gaston_labels=np.max(gaston_labels)-gaston_labels
    
    return gaston_isodepth, gaston_labels
    
def infer_knngraph(coords_mat, cell_type_df=None, n_neighbors=7, plot=False):
    
    start = time.perf_counter()
    
    points = coords_mat  # points to construct graph
    matches = 0  # number of neighbors that are same cell type
    total = 0  # total connections formed
    
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(points)  # create k-NN
    distances, indices = nbrs.kneighbors(points)

    if plot:
        plt.scatter(points[:, 0], points[:, 1], c='blue', label='Points', s=3)
        plt.autoscale()

        # plt.xlim(1250, 3300)
        # plt.ylim(1250, 3300)
        
    for i in range(points.shape[0]):
        for j in indices[i]:
            if i != j:  # Avoid self-connections
                if plot:
                    plt.plot([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]], 'k-', alpha=0.6)
                if cell_type_df is not None and (cell_type_df.iloc[i] == cell_type_df.iloc[j]).iloc[0]:  # determine if neighbors are same cell type
                    matches += 1
                total += 1
    
    if plot:
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title(f'{n_neighbors}-Nearest Neighbors Graph')

    if cell_type_df is not None:
        accuracy = (matches / total) * 100  # calculate pro portion of neighbors that are same cell type
        print(f"Accuracy for {n_neighbors}-Nearest Neighbors is {accuracy}%")
    
    end = time.perf_counter()
    print(f"Graph construction completed in {end - start:0.4f} seconds")
    
    knn_graph = nbrs.kneighbors_graph(mode='connectivity').astype(float)

    return knn_graph


def dag_orient_edges(adjacency_matrix, isodepth):
    if adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
        raise ValueError("adjacency_matrix must be square")
    if adjacency_matrix.shape[0] != len(isodepth):
        raise ValueError("Length of isodepth must match dimensions of adjacency_matrix")
    
    A = adjacency_matrix.astype(bool).astype(float)
    D = -1 * np.sign(isodepth[:, None] - isodepth).T
    D = (D == 1).astype(float)
    D = (A.toarray() * D).astype(bool).astype(float)
    return D


def construct_dag(adata, isodepth, coords_mat, cell_type_df=None, n_neighbors=7, plot=False):

    # Check if cell_type_df is provided and adjust the function call accordingly
    knn_graph = infer_knngraph(coords_mat=coords_mat, cell_type_df=cell_type_df, n_neighbors=n_neighbors, plot=plot)

    dag_adjacency_matrix = dag_orient_edges(knn_graph, isodepth).astype(float)
    
    adata.uns["DAG"] = dag_adjacency_matrix

    return dag_adjacency_matrix



from scipy.spatial import cKDTree
from scipy.interpolate import griddata

def get_shrink_param(marker_size):
    """
    marker_size is in points^2 for each scatter point.
    We'll take sqrt(marker_size) as an approximate radius in 'points'.
    """
    return np.sqrt(marker_size) / 2  # divide by 2 to get radius from diameter

def plot_spatial_dag(dag_adjacency_matrix, coords_mat, isodepth=None, 
                     isodepth_levels=None, file=None, 
                     size=50, arrow_size=0.2, lims=None, 
                     legend=False, color="Reds", title="DAG"):
    df = pd.DataFrame(dag_adjacency_matrix)

    # Extract edges
    rows, cols = np.where(df == 1)
    dag_adjacency_arr = [(row, df.columns[col]) for row, col in zip(rows, cols)]

    points = coords_mat

    plt.figure(figsize=(30, 30))
    ax = plt.gca()

    # Main scatter plot
    scatter = plt.scatter(
        points[:, 0], points[:, 1], 
        c=isodepth if isodepth is not None else 'blue', 
        cmap=color if isodepth is not None else None, 
        s=size, edgecolors='black', linewidths=0.8, alpha=0.9
    )

    # Optional colorbar
    if legend and isodepth is not None:
        cbar = plt.colorbar(scatter)
        cbar.set_label('Isodepth', fontsize=20)

    # Adjust arrow size for visual clarity
    arrow_scale = np.sqrt(size) * arrow_size  
    shrink_val = get_shrink_param(size)  # "shrink" for arrow

    if isodepth_levels is None:
        for i, j in dag_adjacency_arr:
            arrow = FancyArrowPatch(
                (points[i, 0], points[i, 1]), 
                (points[j, 0], points[j, 1]),
                arrowstyle=f'->,head_width={arrow_scale},head_length={arrow_scale * 3}',
                linewidth=4,
                color='black',
                alpha=0.9,
                # The magic:
                shrinkA=shrink_val,  # Shrink arrow tail
                shrinkB=shrink_val,  # Shrink arrow head
            )
            ax.add_patch(arrow)
    else:
        # Code for plotting contours etc.
        ...
        # (unchanged from your original, omitted for brevity)

    if lims:
        ax.set_xlim(lims[0])  # x-range
        ax.set_ylim(lims[1])  # y-range

    plt.title(title, fontsize=24)
    
    plt.axis("off")
    if file:
        plt.savefig(f"{file}.png", dpi=300, bbox_inches='tight')

    plt.show()


    
def pw_fit_dict(counts_mat, gaston_labels, gaston_isodepth, cell_type_df, ct_list, zero_fit_threshold=75,
                t=0.1,umi_threshold=500, isodepth_mult_factor=0.1):
    
    pw_fit_dict=segmented_fit.pw_linear_fit(counts_mat, gaston_labels, gaston_isodepth, 
                                          cell_type_df, ct_list, zero_fit_threshold=zero_fit_threshold, t=t,umi_threshold=umi_threshold,
                                       isodepth_mult_factor=isodepth_mult_factor)
    return pw_fit_dict

def binning_output(counts_mat, gene_labels, gaston_labels, gaston_isodepth, cell_type_df, ct_list, zero_fit_threshold=75,
                t=0.1,umi_threshold=500, isodepth_mult_factor=0.1, num_bins_per_domain=[5,10,5,5]):
    
        pw_fit_dict=segmented_fit.pw_linear_fit(counts_mat, gaston_labels, gaston_isodepth, 
                                          cell_type_df, ct_list, zero_fit_threshold=zero_fit_threshold, t=t,umi_threshold=umi_threshold,
                                       isodepth_mult_factor=isodepth_mult_factor)
        
        binning_output=binning_and_plotting.bin_data(counts_mat.T, gaston_labels, gaston_isodepth, 
                         cell_type_df, gene_labels, num_bins_per_domain=num_bins_per_domain, umi_threshold=umi_threshold)
        
        return pw_fit_dict, binning_output
        
def target_list_old(pw_fit_dict, binning_output, size = 100):
    
    targets = binning_output['gene_labels_idx'][ np.argsort(np.max(np.abs(pw_fit_dict['all_cell_types'][0]),1)) ][::-1][:size]

    return targets

def target_list(pw_fit_dict, binning_output, q, file = None):
    
    discont_genes_layer=spatial_gene_classification.get_discont_genes(pw_fit_dict, binning_output,q=q) #create gene layer dictionaries
    cont_genes_layer=spatial_gene_classification.get_cont_genes(pw_fit_dict, binning_output,q=q)
    
    genes = list(discont_genes_layer.keys()) + list(cont_genes_layer.keys()) #convert keys into one array

    targets = set(genes)
    
    if file is not None:
        with open(file, 'r') as file: #open file of tfs and convert to array
            receptors = file.readlines()
        receptors = [line.strip() for line in receptors]
        receptor_set = set(receptors)
        targets = targets.intersection(receptor_set)

    
    return targets

# def reg_list(file, pw_fit_dict, binning_output, q):
    
#     with open(file, 'r') as file: #open file of tfs and convert to array
#         tfs = file.readlines()
#     tfs = [line.strip() for line in tfs]
    
#     discont_genes_layer=spatial_gene_classification.get_discont_genes(pw_fit_dict, binning_output,q=q) #create gene layer dictionaries
#     cont_genes_layer=spatial_gene_classification.get_cont_genes(pw_fit_dict, binning_output,q=q)
    
#     genes = list(discont_genes_layer.keys()) + list(cont_genes_layer.keys()) #convert keys into one array

#     tf_set = set(tfs)
#     gene_set = set(genes)
    
#     reg_set = tf_set.intersection(gene_set) #check intersection to create list of all TFs
    
#     return reg_set
    

def is_target(adata, targets):
    targets_set = set(targets)
    adata.var['is_target'] = adata.var.index.isin(targets_set)

def is_reg(adata, reg_set):
    adata.var['is_reg'] = adata.var.index.isin(reg_set)

def create_tf_targets(adata, pw_fit_dict, binning_output, q=0.7, file = 'transcription_factors.txt'):
    targets = target_list(pw_fit_dict, binning_output, q, file = None)
    reg_set = target_list(pw_fit_dict, binning_output, q, file = file)
    is_target(adata, targets)
    is_reg(adata, reg_set)

def create_lr_targets(adata, pw_fit_dict, binning_output, q=0.7, file1 = 'receptor.txt', file2 = 'ligand.txt'):
    receptors = target_list(pw_fit_dict, binning_output, q, file = file1)
    ligands = target_list(pw_fit_dict, binning_output, q, file = file2)
    is_target(adata, receptors)
    is_reg(adata, ligands)
    
def run_with_presets():
    return 0

  