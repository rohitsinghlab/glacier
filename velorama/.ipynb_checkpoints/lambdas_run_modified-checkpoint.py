#!/usr/bin/env python

### Authors: Anish Mudide (amudide), Alex Wu (alexw16), Rohit Singh (rs239)
### 2022
### MIT Licence
###

import os
import numpy as np
import scanpy as sc
import argparse
import time
import ray
from ray import tune
import statistics
import scvelo as scv
import pandas as pd
import shutil

from models import *
from train import *
from utils import *

def is_nan_like(x):
    return x is None or x != x or x == ''

def convert_doubles_to_floats(data):
    # Check for NaN values and convert data types
    if isinstance(data, np.ndarray):
        nan_count = np.isnan(data).sum()
        if data.dtype == 'float64':
            data = data.astype('float32')
    elif isinstance(data, pd.DataFrame):
        nan_count = data.isna().sum().sum()
        data = data.astype('float32')
    else:
        nan_count = 0
    
    return data, nan_count

def convert_adata_to_float(adata):
    total_nan_count = 0
    
    for key in adata.uns:
        adata.uns[key], nan_count = convert_doubles_to_floats(adata.uns[key])
        total_nan_count += nan_count
    
    for key in adata.obs:
        adata.obs[key], nan_count = convert_doubles_to_floats(adata.obs[key])
        total_nan_count += nan_count
    
    for key in adata.var:
        adata.var[key], nan_count = convert_doubles_to_floats(adata.var[key])
        total_nan_count += nan_count
    
    for key in adata.obsm:
        adata.obsm[key], nan_count = convert_doubles_to_floats(adata.obsm[key])
        total_nan_count += nan_count
    
    for key in adata.varm:
        adata.varm[key], nan_count = convert_doubles_to_floats(adata.varm[key])
        total_nan_count += nan_count
    
    for key in adata.layers:
        adata.layers[key], nan_count = convert_doubles_to_floats(adata.layers[key])
        total_nan_count += nan_count
    
    # Convert the main X matrix if it's a numpy array
    if isinstance(adata.X, np.ndarray):
        nan_count = np.isnan(adata.X).sum()
        if adata.X.dtype == 'float64':
            adata.X = adata.X.astype('float32')
        total_nan_count += nan_count
    elif isinstance(adata.X, pd.DataFrame):
        nan_count = adata.X.isna().sum().sum()
        adata.X = adata.X.astype('float32')
        total_nan_count += nan_count
    
    print(f"Total NaN count: {total_nan_count}")
    return adata

    

def execute_cmdline():

	parser = argparse.ArgumentParser()
	parser.add_argument('-n','--name',dest='name',type=str,default='velorama_run',help='substring to have in our output files')
	#added default
	# parser.add_argument('-ds','--dataset',dest='dataset', default = 'dataset_A',type=str)
	# parser.add_argument('-ds','--dataset',dest='dataset', default = 'cerebellum_layer_0_q8',type=str)
	# parser.add_argument('-dyn','--dyn',dest='dynamics',type=str,default='pseudotime', 
	# 					choices=['pseudotime','rna_velocity','pseudotime_time','pseudotime_precomputed', 'dag_precomputed'])
	parser.add_argument('-dyn','--dyn',dest='dynamics',type=str,default='dag_precomputed', 
						choices=['pseudotime','rna_velocity','pseudotime_time','pseudotime_precomputed', 'dag_precomputed'])
	parser.add_argument('-ptloc','--ptloc',dest='ptloc',type=str,default='pseudotime')
	parser.add_argument('-dev','--device',dest='device',type=str,default='cuda')
	parser.add_argument('-s','--seed',dest='seed',type=int,default=0,help='Random seed. Set to 0,1,2 etc.')
	parser.add_argument('-lmr','--lam_ridge',dest='lam_ridge',type=float,default=0., help='Currenty unsupported')
	parser.add_argument('-p','--penalty',dest='penalty',type=str,default='H')
	# parser.add_argument('-l','--lag',dest='lag',type=int,default=5)
	
	parser.add_argument('-hd', '--hidden',dest='hidden',type=int,default=32)
	# parser.add_argument('-mi','--max_iter',dest='max_iter',type=int,default=1000)
	
	
	parser.add_argument('-pr','--proba',dest='proba',type=int,default=1)
	parser.add_argument('-ce','--check_every',dest='check_every',type=int,default=10)
	#added default
	parser.add_argument('-rd','--root_dir',dest='root_dir',type=str, default = "./datasets")
	parser.add_argument('-sd','--save_dir',dest='save_dir',type=str,default="./results")
	parser.add_argument('-ls','--lam_start',dest='lam_start',type=float,default=-2)
	parser.add_argument('-le','--lam_end',dest='lam_end',type=float,default=1)
	parser.add_argument('-xn','--x_norm',dest='x_norm',type=str,default='none') # ,choices=['none','zscore','to_count:zscore','zscore_pca','maxmin','fill_zscore'])
	# parser.add_argument('-xn','--x_norm',dest='x_norm',type=str,default='zscore')
	# parser.add_argument('-nl','--num_lambdas',dest='num_lambdas',type=int,default=19)
	
	parser.add_argument('-rt','--reg_target',dest='reg_target',type=int,default=1)
	parser.add_argument('-nn','--n_neighbors',dest='n_neighbors',type=int,default=30)
	parser.add_argument('-vm','--velo_mode',dest='velo_mode',type=str,default='stochastic')
	parser.add_argument('-ts','--time_series',dest='time_series',type=int,default=0)
	parser.add_argument('-nc','--n_comps',dest='n_comps',type=int,default=50)


	parser.add_argument('-mi','--max_iter',dest='max_iter',type=int,default=100)
	parser.add_argument('-lr','--learning_rate',dest='learning_rate',type=float,default=0.1)
	parser.add_argument('-ds','--dataset',dest='dataset', default = 'cerebellum_layer_0_q8',type=str)
	parser.add_argument('-nl','--num_lambdas',dest='num_lambdas',type=int,default=5)
	parser.add_argument('-l','--lag',dest='lag',type=int,default=5)
	parser.add_argument('-sn','--save_name',dest='save_name',type=str,default="lambdas_that_works")


	args = parser.parse_args()

	if not os.path.exists(args.save_dir):
		os.mkdir(args.save_dir)

	adata = sc.read(os.path.join(args.root_dir,'{}.h5ad'.format(args.dataset)))

	adata = adata#[:2000].copy()
	adata.uns["DAG"] = adata.uns["DAG"]#[:2000,:2000]
    
	print("Flag 343.01 ", adata.shape, adata.uns['DAG'].shape) 
#	import sys; sys.exit(0)
    
	if not args.reg_target:
		adata.var['is_target'] = True
		adata.var['is_reg'] = True

	target_genes = adata.var.index.values[adata.var['is_target']]
	reg_genes = adata.var.index.values[adata.var['is_reg']]

	if args.x_norm == 'zscore':

		print('Normalizing data: 0 mean, 1 SD')
		X_orig = adata[:,adata.var['is_reg']].X.toarray().copy()
		std = X_orig.std(0)
		std[std == 0] = 1
		X = torch.FloatTensor(X_orig-X_orig.mean(0))/std
		if 'De-noised' not in args.dataset:
			X = torch.clip(X,-5,5)

		Y_orig = adata[:,adata.var['is_target']].X.toarray().copy()
		std = Y_orig.std(0)
		std[std == 0] = 1
		Y = torch.FloatTensor(Y_orig-Y_orig.mean(0))/std
		if 'De-noised' not in args.dataset:
			Y = torch.clip(Y,-5,5)

	elif args.x_norm == 'magic_zscore':

		import magic
		from scipy.sparse import issparse

		X = adata.X.toarray() if issparse(adata.X) else adata.X 
		X = pd.DataFrame(X,columns=adata.var.index.values)
		magic_operator = magic.MAGIC()
		X_magic = magic_operator.fit_transform(X).astype(np.float32)

		X_orig = X_magic.values[:,adata.var['is_reg'].values]
		std = X_orig.std(0)
		std[std == 0] = 1
		X = torch.FloatTensor(X_orig-X_orig.mean(0))/std
		# X = torch.clip(X,-5,5)

		Y_orig = X_magic.values[:,adata.var['is_target'].values]
		std = Y_orig.std(0)
		std[std == 0] = 1
		Y = torch.FloatTensor(Y_orig-Y_orig.mean(0))/std

	elif args.x_norm == 'fill_zscore':
		X_orig = adata[:,adata.var['is_reg']].X.toarray().copy()
		X_df = pd.DataFrame(X_orig)
		X_df[X_df < 1e-9] = np.nan
		X_df = X_df.fillna(X_df.median())
		X_orig = X_df.values
		std = X_orig.std(0)
		std[std == 0] = 1
		X = torch.FloatTensor(X_orig-X_orig.mean(0))/std
		# X = torch.clip(X,-5,5)

		Y_orig = adata[:,adata.var['is_target']].X.toarray().copy()
		Y_df = pd.DataFrame(Y_orig)
		Y_df[Y_df < 1e-9] = np.nan
		Y_df = Y_df.fillna(Y_df.median())
		Y_orig = Y_df.values
		std = Y_orig.std(0)
		std[std == 0] = 1
		Y = torch.FloatTensor(Y_orig-Y_orig.mean(0))/std
		# Y = torch.clip(Y,-5,5)


	elif args.x_norm == 'to_count:zscore':

		print('Use counts: 0 mean, 1 SD')
		X_orig = adata[:,adata.var['is_reg']].X.toarray().copy()
		X_orig = 2**X_orig-1
		std = X_orig.std(0)
		std[std == 0] = 1
		X = torch.FloatTensor(X_orig-X_orig.mean(0))/std

		Y_orig = adata[:,adata.var['is_target']].X.toarray().copy()
		Y_orig = 2**Y_orig-1
		std = Y_orig.std(0)
		std[std == 0] = 1
		Y = torch.FloatTensor(Y_orig-Y_orig.mean(0))/std

	elif args.x_norm == 'zscore_pca':

		print('PCA + normalizing data: 0 mean, 1 SD')

		sc.tl.pca(adata,n_comps=100)
		adata.X = adata.obsm['X_pca'].dot(adata.varm['PCs'].T)
		X_orig = adata[:,adata.var['is_reg']].X.toarray().copy()
		std = X_orig.std(0)
		std[std == 0] = 1
		X = torch.FloatTensor(X_orig-X_orig.mean(0))/std
		X = torch.clip(X,-5,5)

		Y_orig = adata[:,adata.var['is_target']].X.toarray().copy()
		std = Y_orig.std(0)
		std[std == 0] = 1
		Y = torch.FloatTensor(Y_orig-Y_orig.mean(0))/std	
		Y = torch.clip(Y,-5,5)	

	elif args.x_norm == 'maxmin':

		X_orig = adata[:,adata.var['is_reg']].X.toarray().copy()
		X_min = X_orig.min(0)
		X_max = X_orig.max(0)
		X = torch.FloatTensor((X_orig-X_min)/(X_max-X_min))
		X -= X.mean(0)

		Y_orig = adata[:,adata.var['is_target']].X.toarray().copy()
		Y_min = Y_orig.min(0)
		Y_max = Y_orig.max(0)
		Y = torch.FloatTensor((Y_orig-Y_min)/(Y_max-Y_min))
		Y -= Y.mean(0)

	else:
		assert args.x_norm == 'none'				
		X = torch.FloatTensor(adata[:,adata.var['is_reg']].X.toarray())
		Y = torch.FloatTensor(adata[:,adata.var['is_target']].X.toarray())

	print('# of Regs: {}, # of Targets: {}'.format(X.shape[1],Y.shape[1]))
    
	adata = convert_adata_to_float(adata)
    
	print('AnnData values converted to float')
    
	print('Constructing DAG...')

	if 'De-noised' in args.dataset:
		sc.pp.normalize_total(adata, target_sum=1e4)
		sc.pp.log1p(adata)

	print("Flag 343.07 ", pd.DataFrame(adata.X).iloc[:5,:5])

	sc.pp.scale(adata)

    
	print("Flag 343.09 ", pd.DataFrame(adata.X).iloc[:5,:5])
    
	gmax= np.max(adata.X, axis=0)
	gstd= np.std(adata.X, axis=0)
	gmin= np.min(adata.X, axis=0)
	gmu= np.sum(adata.X, axis=0)
	print("Flag 343.10 ", adata.shape, len(gmu), gmu[:5])
	print("Flag 343.11 ", adata.shape, len(gmax), gmax[:5])
	print("Flag 343.12 ", adata.shape, len(gmin), gmin[:5])
	print("Flag 343.13 ", adata.shape, len(gstd), gstd[:5], np.sum(gstd < 1e-5))
    
	to_remove = gstd < 1e-5
	adata = adata[:,~to_remove].copy()
    
	gstd= np.std(adata.X, axis=0)
	print("Flag 343.15 ", adata.shape, len(gstd), gstd[:5], np.sum(gstd < 1e-5))
    
    #check for nans
    
	if args.dynamics == 'dag_precomputed':
		A = adata.uns['DAG']

	else: 
		A = construct_dag(adata,dynamics=args.dynamics,ptloc=args.ptloc,proba=args.proba,
                          n_neighbors=args.n_neighbors,velo_mode=args.velo_mode,
                          use_time=args.time_series,n_comps=args.n_comps)
#check for nans

	A = torch.FloatTensor(A)
	print("Flag 343.30 ", A.shape, torch.isnan(A).sum().item())

	AX = calculate_diffusion_lags(A,X,args.lag)
	print("Flag 343.31 ", AX.shape, torch.isnan(AX).sum().item())

#check for nans

    
	if args.reg_target:
		AY = calculate_diffusion_lags(A,Y,args.lag)
	else:
		AY = None


	print("Flag 343.32 ", AY.shape, torch.isnan(AY).sum().item())
	print("Flag 343.33 ", X.shape, torch.isnan(X).sum().item())
# 	import sys; sys.exit(0)

        
	dir_name = '{}.seed{}.h{}.{}.lag{}.{}'.format(args.name,args.seed,args.hidden,args.penalty,args.lag,args.dynamics)

	if not os.path.exists(os.path.join(args.save_dir,dir_name)):
		os.mkdir(os.path.join(args.save_dir,dir_name))

	# change number of cpus
    
    
#check for nans

	
	total_start = time.time()
	lam_list = np.logspace(args.lam_start, args.lam_end, num=2).tolist()

	

	# Iterative tuning loop
	stop_condition_met = False
	iteration = 0
	ray.init(object_store_memory=16*10**9, num_cpus=4)
	all_l1s = []
	all_l0s = []
	run_all_time = False
	lambdas_that_work = []
	while not stop_condition_met:
		print(f"Iteration {iteration}, current lam_list: {lam_list}")
		config = {
			'name': args.name,
			'AX': AX,
			'AY': AY,
			'Y': Y,
			'seed': args.seed,
			'lr': args.learning_rate,
			'lam': tune.grid_search(lam_list),
			'lam_ridge': args.lam_ridge,
			'penalty': args.penalty,
			'lag': args.lag,
			'hidden': [args.hidden],
			'max_iter': args.max_iter,
			'device': args.device,
			'lookback': 5,
			'check_every': args.check_every,
			'verbose': True,
			'dynamics': args.dynamics,
			'results_dir': args.save_dir,
			'dir_name': dir_name,
			'reg_target': args.reg_target
		}
		
		resources_per_trial = {"cpu": 1, "gpu": 0.1, "memory": 4 * 1024 * 1024 * 1024}

		# Run the chunk of code
		analysis = tune.run(
			train_model,
			resources_per_trial=resources_per_trial,
			config=config,
			local_dir=os.path.join(args.save_dir)
		)

		# Collect results
		target_folder = os.path.join(args.save_dir, dir_name)
		for subdir, dirs, files in os.walk(args.save_dir):
			for file in files:
				if '.pt' in file:
					file_path = os.path.join(subdir, file)
					target_path = os.path.join(target_folder, file)
					if file_path != target_path:
						shutil.copy(file_path, target_path)
						print(f"Copied {file_path} to {target_path}")

		# Aggregate results
		print("Aggregating results...")
		lam_list = [np.round(lam, 20) for lam in lam_list]
		all_lags = load_gc_interactions(
			args.name, args.save_dir, lam_list,
			hidden_dim=args.hidden, lag=args.lag,
			penalty=args.penalty, dynamics=args.dynamics,
			seed=args.seed, ignore_lag=False
		)
		if run_all_time:
			print("Flag 343.80 ", len(all_lags))
			gc_mat = estimate_interactions(all_lags,lag=args.lag)
			print("Flag 343.81 ", len(gc_mat)) #all_lags))
			gc_df = pd.DataFrame(gc_mat.cpu().data.numpy(),index=target_genes,columns=reg_genes)
			gc_df.to_csv(os.path.join(args.save_dir,'{}.{}.velorama.interactions.tsv'.format(args.name,args.dynamics)),sep='\t')

			lag_mat = estimate_lags(all_lags,lag=args.lag)
			lag_df = pd.DataFrame(lag_mat.cpu().data.numpy(),index=target_genes,columns=reg_genes)
			lag_df.to_csv(os.path.join(args.save_dir,'{}.{}.velorama.lags.tsv'.format(args.name,args.dynamics)),sep='\t')

			print('Total time:',time.time()-total_start)
			np.savetxt(os.path.join(args.save_dir,dir_name + '.time.txt'),np.array([time.time()-total_start]))
			stop_condition_met = True
			break

		first = []
		second = []

		# Check stop condition
		for i in range(len(all_lags)):
			for j in range(args.lag):
				mean_val = (all_lags[i, :, :, j] != 0).float().mean().item()
				if not i:
					first.append(mean_val)
				else:
					second.append(mean_val)
				


				print(f"Mean value for all_lags[{i}, :, :, {j}]: {mean_val}")

				if 0.1 < mean_val < 0.95:
					lambdas_that_work.append(lam_list[i])
			
		print(f"\n\nlower lambda: {first},\nupper lambda: {second}\n\n")

			
					
		if any(0.1 < x < 0.95 for x in first) and any(0.1 < x < 0.95 for x in second):
			run_all_time = True
			lambdas_that_work = np.unique(np.array(lambdas_that_work))
			lam_list = np.linspace(lambdas_that_work.min(), lambdas_that_work.max(), num=args.num_lambdas).tolist()
			

		elif (np.max(first) >= 0.95) and (np.max(second) <= 0.1):
			all_l1s.append(lam_list[1])
			lam_list[1] = (lam_list[0] + lam_list[1]) / 2
			
			print(f"Adjusting lam_list[1] to {lam_list[1]} for the next iteration.")
			iteration += 1
		elif (np.max(first) >= 0.95) and (np.max(second) >= 0.95):
			lam_list[0] = lam_list[1]
			all_l0s.append(lam_list[0])
			lam_list[1] = (all_l1s[-1] + lam_list[0])/2
			print(f"Adjusting lam_list[0] to {lam_list[0]} for the next iteration.")
			print(f"Adjusting lam_list[1] to {lam_list[1]} for the next iteration.")
			iteration += 1
		
		else:
			if (np.max(first) >= 0.95):
				lam_list[0] = (lam_list[0] + lam_list[1])/2
			if (np.max(second) <= 0.1):
				lam_list[1] = (lam_list[0] + lam_list[1])/2

	lambdas_that_work = np.unique(np.array(lambdas_that_work))
	print(f"\n\n\nTHESE ARE THE LAMBDAS THAT WORK {lambdas_that_work}\n\n\n")
	# np.save(args.save_name,lambdas_that_work)

	
	




if __name__ == "__main__":
	execute_cmdline()
