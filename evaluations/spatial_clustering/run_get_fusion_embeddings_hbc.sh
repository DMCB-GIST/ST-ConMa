#!/bin/bash

# Extract fusion embeddings from trained ST_AlignmentModel (Human Breast Cancer)
# Saves embeddings to results directory

# Configuration
CHECKPOINT="./results/spatial_clustering/st_conma/human_breast_cancer/checkpoints"
OUTPUT_DIR="./results/spatial_clustering/st_conma/human_breast_cancer/fusion_embeddings_test"

# Model settings (must match checkpoint)
VISION_MODEL="pathoduet"
VISION_DIM=768
GENE_MODEL="pythia_410m"
GENE_DIM=1024
NUM_LAYERS=12
PROJ_DIM=768
PROJ_HIDDEN_DIM=3072
PROJ_LAYERS=2
NUM_CROSS_LAYERS=3
NUM_HEADS=12

BATCH_SIZE=32
NUM_WORKERS=8
MAX_LENGTH=512

# Output name
OUTPUT_NAME="fusion_embeddings.npy"

echo "=================================="
echo "Extract Fusion Embeddings (Human Breast Cancer)"
echo "=================================="
echo "Checkpoint Directory: $CHECKPOINT"
echo "Output Directory: $OUTPUT_DIR"
echo "Vision Model: $VISION_MODEL ($VISION_DIM dim)"
echo "Gene Model: $GENE_MODEL ($GENE_DIM dim, $NUM_LAYERS layers)"
echo "Cross-attention Layers: $NUM_CROSS_LAYERS"
echo "Output Name: $OUTPUT_NAME"
echo "=================================="

# Run extraction
python ./evaluations/spatial_clustering/get_fusion_embeddings.py \
    --dataset human_breast_cancer \
    --checkpoint $CHECKPOINT \
    --output_dir $OUTPUT_DIR \
    --vision_model $VISION_MODEL \
    --vision_dim $VISION_DIM \
    --gene_model $GENE_MODEL \
    --gene_dim $GENE_DIM \
    --num_layers $NUM_LAYERS \
    --proj_dim $PROJ_DIM \
    --proj_hidden_dim $PROJ_HIDDEN_DIM \
    --proj_layers $PROJ_LAYERS \
    --num_cross_layers $NUM_CROSS_LAYERS \
    --num_heads $NUM_HEADS \
    --max_length $MAX_LENGTH \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --device cuda:3 \
    --output_name $OUTPUT_NAME

echo "=================================="
echo "Extraction completed!"
echo "=================================="
