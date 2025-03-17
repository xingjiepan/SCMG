import json
import warnings
import pkg_resources

import numpy as np
import pandas as pd
import scipy.sparse

import anndata
import scanpy as sc


class GeneNameMapper:
    '''A class to map gene names and IDs for mouse and human.'''
    def __init__(self, 
                 human_gene_csv=None, 
                 mouse_gene_csv=None, 
                 human_mouse_orthologues_csv=None, 
                 mouse_human_orthologues_csv=None,
                 ):
        
        # Define the input files
        if human_gene_csv is None:
            human_gene_csv = pkg_resources.resource_filename('scmg', 
                            'data/gene_names/human_genes.csv')
        if mouse_gene_csv is None:
            mouse_gene_csv = pkg_resources.resource_filename('scmg', 
                            'data/gene_names/mouse_genes.csv')
        if human_mouse_orthologues_csv is None:
            human_mouse_orthologues_csv = pkg_resources.resource_filename('scmg', 
                            'data/gene_names/orthologue_map_human2mouse_best.csv')
        if mouse_human_orthologues_csv is None:
            mouse_human_orthologues_csv = pkg_resources.resource_filename('scmg',
                            'data/gene_names/orthologue_map_mouse2human_best.csv')
    
        # Load the data frames
        human_gene_df = pd.read_csv(human_gene_csv, index_col=0)
        human_gene_df = human_gene_df[~human_gene_df['gene_name'].isna()]
        
        mouse_gene_df = pd.read_csv(mouse_gene_csv, index_col=0)
        mouse_gene_df = mouse_gene_df[~mouse_gene_df['gene_name'].isna()]
        
        human_mouse_df = pd.read_csv(human_mouse_orthologues_csv, index_col=0)
        mouse_human_df = pd.read_csv(mouse_human_orthologues_csv, index_col=0)
        
        # Create dictionaries for mapping
        self.d_human_id_to_name = {human_gene_df.index[i] : human_gene_df['gene_name'].iloc[i]
                                  for i in range(human_gene_df.shape[0])}
        self.d_human_name_to_id = {human_gene_df['gene_name'].iloc[i] : human_gene_df.index[i]
                                  for i in range(human_gene_df.shape[0])}
        
        self.d_mouse_id_to_name = {mouse_gene_df.index[i] : mouse_gene_df['gene_name'].iloc[i]
                                  for i in range(mouse_gene_df.shape[0])}
        self.d_mouse_name_to_id = {mouse_gene_df['gene_name'].iloc[i] : mouse_gene_df.index[i]
                                  for i in range(mouse_gene_df.shape[0])}
        
        self.d_orth_human_to_mouse = {human_mouse_df.index[i] : human_mouse_df['mus_musculus'].iloc[i]
                                     for i in range(human_mouse_df.shape[0])}
        self.d_orth_mouse_to_human = {mouse_human_df.index[i] : mouse_human_df['homo_sapiens'].iloc[i]
                                     for i in range(mouse_human_df.shape[0])}
        
    def map_gene_names(self, genes, source_organism, target_organism, source_type, target_type):
        '''Map gene names or IDs.
        Args:
            genes: The list of genes to be mapped.
            source_organism: The organism of the source genes. Choose from ("human", "mouse").
            traget_organism: The organism of the target genes. Choose from ("human", "mouse").
            source_type: The type of source genes. Choose from ("id", "name").
            target_type: The type of target genes. Choose from ("id", "name").
        '''
        if (source_organism == target_organism) and (source_type == target_type):
            return genes
        
        elif ((source_organism == 'human') and (target_organism == 'human') 
           and (source_type == 'id') and (target_type == 'name')):
            return [self.d_human_id_to_name.get(g, 'na') for g in genes]
        
        elif ((source_organism == 'human') and (target_organism == 'human') 
           and (source_type == 'name') and (target_type == 'id')):
            return [self.d_human_name_to_id.get(g, 'na') for g in genes]
        
        elif ((source_organism == 'mouse') and (target_organism == 'mouse') 
           and (source_type == 'id') and (target_type == 'name')):
            return [self.d_mouse_id_to_name.get(g, 'na') for g in genes]
        
        elif ((source_organism == 'mouse') and (target_organism == 'mouse') 
           and (source_type == 'name') and (target_type == 'id')):
            return [self.d_mouse_name_to_id.get(g, 'na') for g in genes]
        
        elif ((source_organism == 'mouse') and (target_organism == 'human') 
           and (source_type == 'id') and (target_type == 'id')):
            return [self.d_orth_mouse_to_human.get(g, 'na') for g in genes]
        
        elif ((source_organism == 'mouse') and (target_organism == 'human') 
           and (source_type == 'name') and (target_type == 'id')):
            genes = [self.d_mouse_name_to_id.get(g, 'na') for g in genes]
            return [self.d_orth_mouse_to_human.get(g, 'na') for g in genes]
        
        elif ((source_organism == 'mouse') and (target_organism == 'human') 
           and (source_type == 'id') and (target_type == 'name')):
            genes = [self.d_orth_mouse_to_human.get(g, 'na') for g in genes]
            return [self.d_human_id_to_name.get(g, 'na') for g in genes]
        
        elif ((source_organism == 'mouse') and (target_organism == 'human') 
           and (source_type == 'name') and (target_type == 'name')):
            genes = [self.d_mouse_name_to_id.get(g, 'na') for g in genes]
            genes = [self.d_orth_mouse_to_human.get(g, 'na') for g in genes]
            return [self.d_human_id_to_name.get(g, 'na') for g in genes]
        
        elif ((source_organism == 'human') and (target_organism == 'mouse') 
           and (source_type == 'id') and (target_type == 'id')):
            return [self.d_orth_human_to_mouse.get(g, 'na') for g in genes]
        
        elif ((source_organism == 'human') and (target_organism == 'mouse') 
           and (source_type == 'name') and (target_type == 'id')):
            genes = [self.d_human_name_to_id.get(g, 'na') for g in genes]
            return [self.d_orth_human_to_mouse.get(g, 'na') for g in genes]
        
        elif ((source_organism == 'human') and (target_organism == 'mouse') 
           and (source_type == 'id') and (target_type == 'name')):
            genes = [self.d_orth_human_to_mouse.get(g, 'na') for g in genes]
            return [self.d_mouse_id_to_name.get(g, 'na') for g in genes]
        
        elif ((source_organism == 'human') and (target_organism == 'mouse') 
           and (source_type == 'name') and (target_type == 'name')):
            genes = [self.d_human_name_to_id.get(g, 'na') for g in genes]
            genes = [self.d_orth_human_to_mouse.get(g, 'na') for g in genes]
            return [self.d_mouse_id_to_name.get(g, 'na') for g in genes]
        
def standardize_adata(adata, standard_genes=None):
    '''Standardize an AnnData object to a set of standard genes.
    Args:
        adata (AnnData): AnnData object to be standardized. The adata.var.index
                            should be Ensembl IDs.
        standard_genes (list): List of standard genes to be used for standardization.
    '''
    if None is standard_genes:
        standard_gene_csv = pkg_resources.resource_filename('scmg',
                            'data/standard_genes.csv')
        standard_genes = pd.read_csv(standard_gene_csv)['human_id'].values

    if scipy.sparse.issparse(adata.X):
        adata.X = adata.X.toarray()

    current_genes = list(adata.var.index)

    X = np.zeros((adata.shape[0], len(standard_genes)))
    current_indices = []
    standard_indices = []

    for i, g in enumerate(standard_genes):
        if g in current_genes:
            current_indices.append(current_genes.index(g))
            standard_indices.append(i)

    X[:, standard_indices] = adata.X[:, current_indices]

    adata_standard = anndata.AnnData(
        X=X,
        obs=adata.obs.copy(),
        var=pd.DataFrame(index=standard_genes),
    )
    return adata_standard