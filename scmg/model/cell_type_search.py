import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors


class CellTypeSearcher:
    def __init__(self, ref_cell_emb_adata, emb_key='X_scmg'):
        self.ref_cell_emb_adata = ref_cell_emb_adata
        if emb_key not in ref_cell_emb_adata.obsm.keys():
            self.ref_emb = ref_cell_emb_adata.X
        else:
            self.ref_emb = ref_cell_emb_adata.obsm[emb_key]

        # Get the cell types and their counts
        self.cell_types, self.ct_counts = np.unique(
            ref_cell_emb_adata.obs['cell_type'], return_counts=True)

        # Create a matrix that for weighted maps from 
        # reference cells to cell types
        self.ref_cell_to_ct_mtx = np.zeros((ref_cell_emb_adata.shape[0], 
                                       len(self.cell_types)), dtype=np.float32)
        for i, ct in enumerate(self.cell_types):
            self.ref_cell_to_ct_mtx[ref_cell_emb_adata.obs['cell_type'] == ct, 
                               i] = 1 / self.ct_counts[i]

    def search_ref_cell(self, query_emb, project_umap=True, n_jobs=32):
        '''Search for the reference cells that are most similar to
        each query cell.
        Args:
            query_emb (np.array): query embedding of shape (n_cells, n_features)
        '''
        neigh = NearestNeighbors(n_neighbors=1, n_jobs=n_jobs)
        neigh.fit(self.ref_emb)

        # Find the nearest reference cell for each query cell
        dist, ind = neigh.kneighbors(query_emb)
        ind = ind.flatten()
        dist = dist.flatten()

        neighbor_df = pd.DataFrame({
            'ref_cell': self.ref_cell_emb_adata.obs.index[ind],
            'distance': dist,
        })

        if project_umap:
            neighbor_df['umap_x'] = self.ref_cell_emb_adata.obsm[
                                    'X_umap'][ind, 0]
            neighbor_df['umap_y'] = self.ref_cell_emb_adata.obsm[
                                    'X_umap'][ind, 1]

        return neighbor_df

    def search_ref_cell_types(self, query_emb, radius=2, n_jobs=32):
        '''Search for reference cell types that are most similar to 
        the query cells
        Args:
            query_emb (np.array): query embedding of shape (n_cells, n_features)
            radius (float): soft search radius
        '''
        # Calculate the pairwise distances between the query and reference cells
        q_to_r_dist_mtx = pairwise_distances(query_emb, self.ref_emb,
                                    metric='euclidean', n_jobs=n_jobs)

        # Calculate the weights bewteen the query and reference cells
        assert(radius > 0)
        q_to_r_weights = np.exp(-(q_to_r_dist_mtx / radius) ** 2)

        # Calculate the weighted maps from reference cells to cell types
        q_to_ct_mtx = q_to_r_weights @ self.ref_cell_to_ct_mtx

        # Calculate the weights of cell types
        ct_weights = q_to_ct_mtx.mean(axis=0)

        return pd.DataFrame(ct_weights, index=self.cell_types, 
                columns=['weight']).sort_values('weight', ascending=False)
