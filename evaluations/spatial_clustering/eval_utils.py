"""
Evaluation utilities for spatial clustering.
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
import ot
from sklearn import metrics as sk_metrics
from sklearn.preprocessing import StandardScaler


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='norm_emb', random_seed=42):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata


def search_res(adata, n_clusters, method='leiden', use_rep='emb', start=0.1, end=3.0, increment=0.01):
    """
    Search for the resolution parameter that gives approximately n_clusters.
    """
    print('Searching resolution...')
    sc.pp.neighbors(adata, use_rep=use_rep)

    for res in np.arange(start, end, increment):
        if method == 'leiden':
            sc.tl.leiden(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
        elif method == 'louvain':
            sc.tl.louvain(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique())

        if count_unique == n_clusters:
            print(f'Resolution: {res}, Clusters: {count_unique}')
            return res

    print(f'Warning: Could not find exact resolution for {n_clusters} clusters')
    return res


def clustering(adata, n_clusters=7, radius=50, method='mclust',
               start=0.1, end=3.0, increment=0.01, emb_key='emb'):
    """
    Spatial clustering based on learned representation.

    Parameters
    ----------
    adata : AnnData
        AnnData object with embeddings in adata.obsm[emb_key]
    n_clusters : int
        Number of clusters
    radius : int
        Number of neighbors for refinement
    method : str
        Clustering method: 'mclust', 'leiden', or 'louvain'
    start, end, increment : float
        Resolution search parameters for leiden/louvain
    emb_key : str
        Key in adata.obsm containing embeddings

    Returns
    -------
    adata : AnnData
        Updated AnnData with 'domain' in adata.obs
    """
    if method == 'mclust':
        adata = mclust_R(adata, used_obsm=emb_key, num_cluster=n_clusters)
        adata.obs['domain'] = adata.obs['mclust']
    elif method == 'leiden':
        res = search_res(adata, n_clusters, use_rep=emb_key, method=method,
                        start=start, end=end, increment=increment)
        sc.tl.leiden(adata, random_state=0, resolution=res)
        adata.obs['domain'] = adata.obs['leiden']
    elif method == 'louvain':
        res = search_res(adata, n_clusters, use_rep=emb_key, method=method,
                        start=start, end=end, increment=increment)
        sc.tl.louvain(adata, random_state=0, resolution=res)
        adata.obs['domain'] = adata.obs['louvain']

    return adata


def calculate_metrics(adata, pred_key='domain', label_key='ground_truth'):
    """
    Calculate clustering metrics (ARI, NMI).

    Parameters
    ----------
    adata : AnnData
        AnnData object with predictions and ground truth
    pred_key : str
        Key in adata.obs containing predicted labels
    label_key : str
        Key in adata.obs containing ground truth labels

    Returns
    -------
    dict : Dictionary containing ARI and NMI scores
    """
    if label_key not in adata.obs.columns:
        print(f"Warning: {label_key} not found in adata.obs")
        return {'ari': None, 'nmi': None}

    pred = adata.obs[pred_key]
    label = adata.obs[label_key]

    ari = sk_metrics.adjusted_rand_score(pred, label)
    nmi = sk_metrics.normalized_mutual_info_score(pred, label)

    adata.uns['ari'] = ari
    adata.uns['nmi'] = nmi

    print(f'ARI: {ari:.4f}')
    print(f'NMI: {nmi:.4f}')

    return {'ari': ari, 'nmi': nmi}


def draw_spatial(adata, output_dir, sample_id, pred_key='domain', label_key='ground_truth',
                 img_key='hires', spot_size=1.6, dpi=300, prefix=''):
    """
    Draw spatial plot with ground truth (left) and predicted domain (right).

    Parameters
    ----------
    adata : AnnData
        AnnData object with spatial coordinates
    output_dir : str
        Output directory for saving the plot
    sample_id : str
        Sample ID for filename
    pred_key : str
        Key in adata.obs containing predicted labels
    label_key : str
        Key in adata.obs containing ground truth labels
    img_key : str
        Key for image in adata.uns['spatial']
    spot_size : float
        Size of spots in the plot
    dpi : int
        DPI for saving
    prefix : str
        Optional prefix for filename
    """
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    if label_key in adata.obs.columns:
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        sc.pl.spatial(adata, img_key=img_key, size=spot_size, color=label_key,
                      ax=axes[0], show=False, title=label_key)
        sc.pl.spatial(adata, img_key=img_key, size=spot_size, color=pred_key,
                      ax=axes[1], show=False, title=pred_key)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'spatial_{prefix}{sample_id}.png'),
                    dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        sc.pl.spatial(adata, img_key=img_key, size=spot_size, color=pred_key,
                      ax=ax, show=False, title=pred_key)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'spatial_{prefix}{sample_id}.png'),
                    dpi=dpi, bbox_inches='tight')
        plt.close()

    print(f"Saved: {os.path.join(output_dir, f'spatial_{prefix}{sample_id}.png')}")


def draw_umap(adata, output_dir, sample_id, pred_key='domain', label_key='ground_truth',
              emb_key='emb', n_neighbors=15, dpi=300, prefix=''):
    """
    Draw UMAP plot with ground truth (left) and predicted domain (right).

    Parameters
    ----------
    adata : AnnData
        AnnData object with embeddings
    output_dir : str
        Output directory for saving the plot
    sample_id : str
        Sample ID for filename
    pred_key : str
        Key in adata.obs containing predicted labels
    label_key : str
        Key in adata.obs containing ground truth labels
    emb_key : str
        Key in adata.obsm containing embeddings for UMAP
    n_neighbors : int
        Number of neighbors for UMAP computation
    dpi : int
        DPI for saving
    prefix : str
        Optional prefix for filename
    """
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    print('Computing UMAP...')
    sc.pp.neighbors(adata, use_rep=emb_key, n_neighbors=n_neighbors)
    sc.tl.umap(adata)

    if label_key in adata.obs.columns:
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        sc.pl.umap(adata, color=label_key, ax=axes[0], show=False, title=label_key)
        sc.pl.umap(adata, color=pred_key, ax=axes[1], show=False, title=pred_key)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'umap_{prefix}{sample_id}.png'),
                    dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        sc.pl.umap(adata, color=pred_key, ax=ax, show=False, title=pred_key)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'umap_{prefix}{sample_id}.png'),
                    dpi=dpi, bbox_inches='tight')
        plt.close()

    print(f"Saved: {os.path.join(output_dir, f'umap_{prefix}{sample_id}.png')}")


def draw_paga(adata, output_dir, sample_id, pred_key='domain', label_key='ground_truth',
              emb_key='emb', dpi=300, prefix='', title_suffix=''):
    """
    Draw PAGA comparison plot with ground truth and predicted domain.

    Parameters
    ----------
    adata : AnnData
        AnnData object with embeddings and clustering results
    output_dir : str
        Output directory for saving the plot
    sample_id : str
        Sample ID for filename
    pred_key : str
        Key in adata.obs containing predicted labels
    label_key : str
        Key in adata.obs containing ground truth labels
    emb_key : str
        Key in adata.obsm containing embeddings
    dpi : int
        DPI for saving
    prefix : str
        Optional prefix for filename
    title_suffix : str
        Optional suffix for plot title
    """
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    # Ensure neighbors are computed
    if 'neighbors' not in adata.uns:
        sc.pp.neighbors(adata, use_rep=emb_key)

    # Create a copy to avoid modifying original
    adata_paga = adata.copy()

    plt.rcParams["figure.figsize"] = (4, 3)

    # Use ground_truth for PAGA if available, otherwise use pred_key
    groups_key = label_key if label_key in adata_paga.obs.columns else pred_key

    sc.tl.paga(adata_paga, groups=groups_key)
    sc.pl.paga_compare(adata_paga, legend_fontsize=10, frameon=False, size=20,
                      title=f'{sample_id}{title_suffix}',
                      legend_fontoutline=2, show=False,
                      save=f'_{prefix}{sample_id}.png')

    # Move file from scanpy default location to output_dir
    import shutil
    scanpy_figdir = sc.settings.figdir
    src = os.path.join(scanpy_figdir, f'paga_compare_{prefix}{sample_id}.png')
    dst = os.path.join(output_dir, f'paga_{prefix}{sample_id}.png')
    if os.path.exists(src):
        shutil.move(src, dst)

    print(f"Saved: {dst}")


def draw_density(adata, output_dir, sample_id, label_key='ground_truth',
                 emb_key='emb', dpi=300, prefix=''):
    """
    Draw UMAP embedding density plot.

    Parameters
    ----------
    adata : AnnData
        AnnData object with embeddings and UMAP computed
    output_dir : str
        Output directory for saving the plot
    sample_id : str
        Sample ID for filename
    label_key : str
        Key in adata.obs containing ground truth labels
    emb_key : str
        Key in adata.obsm containing embeddings
    dpi : int
        DPI for saving
    prefix : str
        Optional prefix for filename
    """
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    # Ensure UMAP is computed
    if 'X_umap' not in adata.obsm:
        if 'neighbors' not in adata.uns:
            sc.pp.neighbors(adata, use_rep=emb_key)
        sc.tl.umap(adata)

    plt.rcParams["figure.figsize"] = (5, 5)

    if label_key in adata.obs.columns:
        sc.tl.embedding_density(adata, basis='umap', groupby=label_key)
        sc.pl.embedding_density(adata, basis='umap', groupby=label_key, show=False,
                               save=f'_{prefix}{sample_id}.png')

        # Move file from scanpy default location to output_dir
        # scanpy creates: umap_density_{groupby}_{save}
        import shutil
        scanpy_figdir = sc.settings.figdir
        src = os.path.join(scanpy_figdir, f'umap_density_{label_key}__{prefix}{sample_id}.png')
        dst = os.path.join(output_dir, f'density_{prefix}{sample_id}.png')
        if os.path.exists(src):
            shutil.move(src, dst)
            print(f"Saved: {dst}")
        else:
            print(f"Warning: Could not find {src}")
