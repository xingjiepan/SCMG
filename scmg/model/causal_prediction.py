import warnings

from tqdm import tqdm
import numpy as np
import pandas as pd
import scipy.stats
import scipy.spatial

from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process.kernels as kernels



class CausalGenePredictor:
    def __init__(self, adata_pert, gene_stds):
        '''
        Args:
            adata_pert (AnnData): Annotated data object with the perturbation
                shifts.
        '''
        self.standard_genes = adata_pert.var.index.values
        self.gene_stds = np.maximum(gene_stds, 0.1)

        self.adata_pert = adata_pert[
            adata_pert.obs['perturbed_gene'].isin(self.standard_genes)].copy()
        
    def calc_causal_scores(self, mean_exp_shifts):
        '''
        Args:
            exp_shifts (np.ndarray): The mean expression shifts for the cell
                state transition.

        Returns:
            pd.DataFrame: A DataFrame with the causal scores for each perturbation.
        '''
        mean_shift_z = mean_exp_shifts / self.gene_stds

        # Calculate the match score for each perturbation
        pert_match_df = self.adata_pert.obs.copy()
        gene_z_shifts = []
        pert_sims = []

        for i in tqdm(range(self.adata_pert.X.shape[0])):
            pert_vec = self.adata_pert.X[i]
            pert_vec_z = pert_vec / self.gene_stds

            if (np.std(pert_vec) == 0) or (np.std(mean_shift_z) == 0):
                pms = 0
            else:
                pms = 1 - scipy.spatial.distance.cosine(
                    mean_shift_z, pert_vec_z)

            pert_gene = self.adata_pert.obs['perturbed_gene'].iloc[i]
            pert_gene_i = self.adata_pert.var.index.get_loc(pert_gene)

            gene_z_shifts.append(mean_shift_z[pert_gene_i])
            pert_sims.append(pms)

        pert_match_df['gene_shift_z'] = np.clip(gene_z_shifts, -5, 5)
        pert_match_df['pert_sim'] = pert_sims
        pert_match_df['pert_match_score'] = pert_sims * pert_match_df['perturbation_sign']
        pert_match_df['causal_score'] = pert_match_df['pert_match_score'] * pert_match_df['gene_shift_z']

        return pert_match_df
        