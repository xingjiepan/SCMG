import os
import json

from tqdm.auto import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from scmg.model.basic import MLP

torch.set_float32_matmul_precision('high')



def edge_batch_to_Xs(batch, exp_data, dataset_to_cell_idx_map,
                      max_dropout_rate=0.3):

    cell_ref_idx = batch['cell_ref_idx']
    cell_query_idx = batch['cell_query_idx']
    cell_query_datasets = batch['dataset_query']
    
    # Sample the negative cells
    neg_cell_idx = []
    for ref_cell_dataset in batch['dataset_ref']:
        rand_idx = np.random.randint(low=0,
                    high=len(dataset_to_cell_idx_map[ref_cell_dataset]))
        neg_cell_idx.append(dataset_to_cell_idx_map[ref_cell_dataset][rand_idx])

    X_anchor = exp_data[cell_ref_idx]['X_exp']
    X_positive = exp_data[cell_query_idx]['X_exp']
    X_negative = exp_data[neg_cell_idx]['X_exp']

    # Randomly drop out some of the input data
    dropout_rate = np.random.uniform(low=0, high=1)
    X_anchor_with_drop = X_anchor
    X_positive_with_drop = X_positive
    X_negative_with_drop = X_negative
    
    if dropout_rate < max_dropout_rate:
        X_anchor_with_drop = X_anchor * (torch.rand(*X_anchor.shape) > dropout_rate)
        X_positive_with_drop = X_positive * (torch.rand(*X_anchor.shape) > dropout_rate)
        X_negative_with_drop = X_negative * (torch.rand(*X_anchor.shape) > dropout_rate)

    return X_anchor_with_drop, X_positive_with_drop, X_negative_with_drop, X_positive, cell_query_datasets

class CellEmbedder(nn.Module):
    def __init__(self,
                 n_genes,
                 dataset_id_map,
                 ):
        super(CellEmbedder, self).__init__()
        self.dataset_id_map = dataset_id_map
        self.dataset_emb = nn.parameter.Parameter(
            data=torch.randn(len(dataset_id_map), 64), requires_grad=True)

        self.encoder = MLP(n_genes, 512, hidden_dims=(2048, 2048,),
                           dropout_prob=0.0)
        self.decoder = MLP(512 + 64, n_genes, hidden_dims=(1024, 2048,),
                           dropout_prob=0)
    
    def forward(self, X_exp):
        return self.encoder(X_exp)
    
    def decode(self, X_emb, dataset_names):
        dataset_ids = [self.dataset_id_map[dn] for dn in dataset_names]
        dataset_emb = self.dataset_emb[dataset_ids]
        X_emb_cat = torch.cat([X_emb, dataset_emb], dim=1)

        return F.softplus(self.decoder(X_emb_cat))
        
def l2_loss(X_emb):
    return torch.mean(torch.sum(X_emb.pow(2), dim=1))

def contrastive_loss(X_emb_anchor, X_emb_pos, X_emb_neg):
    # Calculate the similarity between the anchor and the neighbors
    dists_pos = torch.sqrt(torch.sum((X_emb_anchor - X_emb_pos).pow(2), 
                                     dim=1) + 1e-8)
    dists_neg = torch.sqrt(torch.sum((X_emb_anchor - X_emb_neg).pow(2), 
                                     dim=1) + 1e-8)
    dist_diff = dists_pos - dists_neg

    # Calculate the contrastive loss
    loss = F.softplus(dist_diff) 
    return torch.mean(loss)

def recon_loss(X_exp, X_pred):
    return torch.sum(torch.square(X_pred - X_exp)) / X_exp.shape[0]

def calc_loss(model, batch, exp_data, dataset_to_cell_idx_map, device,
            contrastive_weight=1000, l2_weight=1e-1, recon_weight=1e-3):
    (X_anchor_with_drop, X_positive_with_drop, X_negative_with_drop, 
     X_positive, cell_query_datasets) = edge_batch_to_Xs(
                                batch, exp_data, dataset_to_cell_idx_map)
    
    X_anchor_with_drop = X_anchor_with_drop.to(device)
    X_positive_with_drop = X_positive_with_drop.to(device)
    X_negative_with_drop = X_negative_with_drop.to(device)
    X_positive = X_positive.to(device)

    X_emb_anchor = model(X_anchor_with_drop)
    X_emb_pos = model(X_positive_with_drop)
    X_emb_neg = model(X_negative_with_drop)

    l2 = l2_loss(X_emb_anchor) + l2_loss(X_emb_pos) + l2_loss(X_emb_neg)
    contrastive1 = contrastive_loss(X_emb_anchor, X_emb_pos, X_emb_neg)
    contrastive2 = contrastive_loss(X_emb_pos, X_emb_anchor, X_emb_neg)
    contrastive = contrastive1 + contrastive2

    X_pos_pred = model.decode(X_emb_pos, cell_query_datasets)
    recon = recon_loss(X_positive, X_pos_pred)

    return (contrastive_weight * contrastive,
            l2_weight * l2, recon_weight * recon)

def train_contrastive_embedder(
        model,
        edge_loader,
        exp_data,
        dataset_to_cell_idx_map,
        num_epochs,
        output_path,
        lr=1e-3,
        loss_history=None,
    ):
    os.makedirs(output_path, exist_ok=True)
    device = next(model.parameters()).device

    if loss_history is None:
        loss_history = {'train_contrastive': [], 'train_l2': [],
                        'train_total': [], 'train_recon': []}

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    pbar = tqdm(range(num_epochs))

    for epoch in pbar:

        for k in loss_history:
            loss_history[k].append([])

        model.train()

        for batch in edge_loader:
            model.zero_grad()
            contrastive, l2, recon = calc_loss(model, batch, exp_data, 
                                   dataset_to_cell_idx_map, device)
            loss = contrastive + l2 + recon

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2)

            optimizer.step()
            loss_history['train_contrastive'][-1].append(contrastive.item())
            loss_history['train_l2'][-1].append(l2.item())
            loss_history['train_recon'][-1].append(recon.item())
            loss_history['train_total'][-1].append(loss.item())


        mean_loss = np.mean(loss_history['train_total'][-1])
        pbar.set_description(f"Train loss: {mean_loss}")

        with open(f'{output_path}/loss_history.json', 'w') as f:
            json.dump(loss_history, f)
            torch.save(model, f'{output_path}/model.pt')
            torch.save(model.state_dict(), f'{output_path}/best_state_dict.pth')


import scipy.sparse
import pandas as pd
import pkg_resources
import anndata
import sklearn.neighbors


def get_Xs_from_anndata(adata, standard_genes):

    # Define the relationships between indices
    standard_genes = list(standard_genes)
    adata_genes = list(adata.var.index)
    common_ids = np.intersect1d(standard_genes, adata_genes)
    common_in_standard_indices = [standard_genes.index(g) for g in common_ids]
    common_in_adata_indices = [adata_genes.index(g) for g in common_ids]

    # Convert the expression data to numpy array
    if scipy.sparse.issparse(adata.X):
        adata_X = adata.X.toarray()
    else:
        adata_X = adata.X
    
    # Extract the expressions
    X = np.zeros((adata.shape[0], len(standard_genes)), dtype=np.float32)
    X[:, common_in_standard_indices] = adata_X[:, common_in_adata_indices]

    # Normalize the expressions
    X = X / (X.sum(axis=1, keepdims=True) + 1e-6) * 1e4
    X = np.log1p(X)

    # Extract the measurement mask
    X_mask = np.zeros((adata.shape[0], len(standard_genes)), dtype=bool)
    X_mask[:, common_in_standard_indices] = True

    return X, X_mask

def embed_adata(model, 
                 adata, 
                 standard_gene_csv=None,
                 batch_size=512,
                 inplace=True
                 ):
    '''Embed the data in an AnnData object using the given model.
    The adata object should have the following attributes:
    - adata.X should be raw counts.
    - adata.var.index should be human gene Ensembl IDs.
    '''
    device = next(model.parameters()).device

    # Load the standard genes
    if standard_gene_csv is None:
        standard_gene_csv = pkg_resources.resource_filename('scmg',
                            'data/standard_genes.csv')
        
    standard_genes = pd.read_csv(standard_gene_csv)['human_id'].values
    adata_cg = adata[:, adata.var.index.isin(standard_genes)].copy()

    N_cells = adata.shape[0]
    N_batches = int(np.ceil(N_cells / batch_size))
    Z = []

    for i in tqdm(range(N_batches)):
        start = i * batch_size
        stop = min((i + 1) * batch_size, N_cells)
        adata_batch = adata_cg[start:stop].copy()

        X, _ = get_Xs_from_anndata(adata_batch, standard_genes)
        X = torch.tensor(X, dtype=torch.float32).to(device)

        with torch.no_grad():
            z = model(X)
            z = z.detach().cpu().numpy()
            for i in range(z.shape[0]):
                Z.append(z[i])

    if inplace:
        adata.obsm['X_ce_latent'] = np.array(Z)
    else:
        return np.array(Z)

def embed_standardized_adata(model,
                              adata,
                              batch_size=512,
                              inplace=True,
                              verbose_level=1
                              ):
    ''' Embed the data in an AnnData object using the given model. The adata
    should be already standardized. The adata object should have the following
    attributes:
    - adata.X should be raw counts.
    - adata.var.index should be the standarded gene set with the same order.
    - adata.layers['measure_mask'] should be the measurement mask.
    '''
    device = next(model.parameters()).device

    N_cells = adata.shape[0]
    N_batches = int(np.ceil(N_cells / batch_size))
    Z = []

    # Decide what to print based on the verbose level
    if verbose_level == 0:
        iterator = range(N_batches)
    else:
        iterator = tqdm(range(N_batches))

    for i in iterator:
        start = i * batch_size
        stop = min((i + 1) * batch_size, N_cells)
        adata_batch = adata[start:stop].copy()

        X = adata_batch.X
        if scipy.sparse.issparse(X):
            X = X.toarray()

        X = X / (X.sum(axis=1, keepdims=True) + 1e-6) * 1e4
        X = np.log1p(X)

        X = torch.tensor(X, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            z = model(X)
            z = z.detach().cpu().numpy()
            for i in range(z.shape[0]):
                Z.append(z[i])

    if inplace:
        adata.obsm['X_ce_latent'] = np.array(Z)
    else:
        return np.array(Z)
    
def decode_adata(model,
                 adata,
                 dataset_names,
                 standard_gene_csv=None,
                 batch_size=512,):
    '''Decode the data in an AnnData object using the given model.
    The input adata object should have adata.obsm['X_ce_latent'].
    Returns a new AnnData object with the decoded gene expression.
    '''
    device = next(model.parameters()).device
    
    # Load the standard genes
    if standard_gene_csv is None:
        standard_gene_csv = pkg_resources.resource_filename('scmg',
                            'data/standard_genes.csv')
    
    standard_genes = pd.read_csv(standard_gene_csv)['human_id'].values

    N_cells = adata.shape[0]
    N_batches = int(np.ceil(N_cells / batch_size))
    X_pred = np.zeros((N_cells, len(standard_genes)), dtype=np.float32)

    for i in tqdm(range(N_batches)):
        start = i * batch_size
        stop = min((i + 1) * batch_size, N_cells)
        Z_batch = adata.obsm['X_ce_latent'][start:stop].copy()
        dataset_names_batch = dataset_names[start:stop]

        X_pred_batch = model.decode(
            torch.tensor(Z_batch, dtype=torch.float32).to(device),
            dataset_names_batch
            ).detach().cpu().numpy()
        X_pred[start:stop] = X_pred_batch

    adata_pred = anndata.AnnData(
        X = X_pred,
        obs = adata.obs.copy(),
        var = pd.DataFrame(index=standard_genes),
        uns = adata.uns.copy(),
        obsm = adata.obsm.copy(),
        obsp = adata.obsp.copy(),
        )
    
    return adata_pred

def decode_cell_state_embedding(
        model,
        X_ce_latent,
        dataset_names,
        standard_gene_csv=None,
        batch_size=512,
):
    '''Decode the given cell state embeddings using the given model.
    Returns an AnnData object with the decoded gene expression.
    '''
    device = next(model.parameters()).device

    # Load the standard genes
    if standard_gene_csv is None:
        standard_gene_csv = pkg_resources.resource_filename('scmg',
                            'data/standard_genes.csv')
    
    standard_genes = pd.read_csv(standard_gene_csv)['human_id'].values

    N_cells = X_ce_latent.shape[0]
    N_batches = int(np.ceil(N_cells / batch_size))
    X_pred = np.zeros((N_cells, len(standard_genes)), dtype=np.float32)

    for i in tqdm(range(N_batches)):
        start = i * batch_size
        stop = min((i + 1) * batch_size, N_cells)
        Z_batch = X_ce_latent[start:stop].copy()
        dataset_names_batch = dataset_names[start:stop]

        X_pred_batch = model.decode(
            torch.tensor(Z_batch, dtype=torch.float32).to(device),
            dataset_names_batch
            ).detach().cpu().numpy()
        X_pred[start:stop] = X_pred_batch

    adata_pred = anndata.AnnData(
        X = X_pred,
        obs = pd.DataFrame(index=np.arange(N_cells).astype(str)),
        var = pd.DataFrame(index=standard_genes),
        obsm = {'X_ce_latent': X_ce_latent},
        )
    
    return adata_pred

def score_marker_genes(model_ce, adata_std_query, target_emb):
    '''Calculate the marker genes that are most important for determining
    the similarity between the query and target datasets.
    Args:
        model_ce: torch.nn.Module
            The trained contrastive embedding model.
        adata_std_query: anndata.AnnData
            The query dataset with the standard gene order and
            raw expression values.
        target_emb: np.ndarray of shape (n_cells, n_features)
            The embedding of the target dataset.
    '''
    neigh = sklearn.neighbors.NearestNeighbors(n_neighbors=1)
    neigh.fit(target_emb)

    embed_standardized_adata(model_ce, adata_std_query)
    dist0 = np.mean(neigh.kneighbors(adata_std_query.obsm['X_ce_latent'], 
                                     return_distance=True)[0])   
    
    dist_dict = {
        'gene': [],
        'dist_shift': []
    }

    # Calcualte the embedding distance shift after
    # masking out each gene
    for i in tqdm(range(adata_std_query.shape[1])):
        gene = adata_std_query.var.index[i]

        adata_query_tmp = adata_std_query.copy()

        if adata_query_tmp.X[:, i].max() == 0:
            dist = dist0
        else:
            adata_query_tmp.X[:, i] = 0
            embed_standardized_adata(model_ce, adata_query_tmp, verbose_level=0)

            dist = np.mean(neigh.kneighbors(adata_query_tmp.obsm['X_ce_latent'], 
                                 return_distance=True)[0])

        dist_dict['gene'].append(gene)
        dist_dict['dist_shift'].append(dist - dist0)

    dist_df = pd.DataFrame(dist_dict).set_index('gene')
    dist_df['dist_shift_z'] = (dist_df['dist_shift'] - dist_df['dist_shift'].mean()
                               ) / dist_df['dist_shift'].std()
    return dist_df