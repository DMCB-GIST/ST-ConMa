"""
STAIG: Spatial Transcriptomics Analysis via Image-Guided Graph Neural Network

This package provides tools for spatial clustering of spatial transcriptomics data
using graph neural networks with optional image guidance.

Modules:
    - staig: Original STAIG model
    - staig_fusion: STAIG with fusion embedding support (dual-view contrastive)
    - net: Network architectures for original STAIG
    - net_fusion: Network architectures for STAIG-Fusion
    - adata_processing: Data loading utilities
    - adata_processing_fusion: Data loading with fusion embedding support
    - utils: Clustering and utility functions
"""

# Lazy imports to avoid rpy2 issues
# from .staig import STAIG  # Commented out - imports metrics.py which requires rpy2
# from .staig_fusion import STAIG_Fusion, run_staig_fusion  # Not implemented yet
from .adata_processing import LoadSingle10xAdata, LoadBatch10xAdata
# from .adata_processing_fusion import (  # Not implemented yet
#     LoadSingle10xAdata_Fusion,
#     LoadSingle10xAdata_FusionOnly,
#     load_dlpfc_with_fusion
# )
from .utils import clustering, mclust_R, refine_label

__version__ = '0.2.0'
__all__ = [
    # 'STAIG',  # Use: from staig.staig import STAIG
    'STAIG_Fusion',
    'run_staig_fusion',
    'LoadSingle10xAdata',
    'LoadBatch10xAdata',
    'LoadSingle10xAdata_Fusion',
    'LoadSingle10xAdata_FusionOnly',
    'load_dlpfc_with_fusion',
    'clustering',
    'mclust_R',
    'refine_label'
]
