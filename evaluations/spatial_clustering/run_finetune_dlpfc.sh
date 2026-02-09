#!/bin/bash

# Fine-tuning Script for DLPFC samples
# Uses outputs from generate_sentences.py and extract_patches.py
# Single GPU, IGC + IGM only (no IIC)


# Configuration
DATA_DIR="./ft_dataset/spatial_clustering/DLPFC"
PRETRAINED_CHECKPOINT="./checkpoints/st_conma_pythia410m_12layers_3aug_clip/checkpoint_epoch_12.pt"
OUTPUT_DIR="./results/spatial_clustering/st_conma/dlpfc/checkpoints"

# Training settings
BATCH_SIZE=32
EPOCHS=1
LR=1e-5
WEIGHT_DECAY=0.01
SEED=42

echo "=================================="
echo "Fine-tuning Configuration"
echo "=================================="
echo "Data Directory: $DATA_DIR"
echo "Pretrained: $PRETRAINED_CHECKPOINT"
echo "Vision Model: pathoduet (768 dim)"
echo "Gene Model: C2S pythia_410m (1024 dim, 12 layers)"
echo "Projection Dim: 768"
echo "Samples: All 12 DLPFC slides"
echo "Loss: IGC + IGM (no IIC)"
echo "Learning Rate: $LR"
echo "Weight Decay: $WEIGHT_DECAY"
echo "Batch Size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "=================================="

# Run fine-tuning
python ./evaluations/spatial_clustering/finetune.py \
    --data_dir $DATA_DIR \
    --use_filtered_images \
    --max_length 512 \
    --vision_model pathoduet \
    --vision_dim 768 \
    --gene_model pythia_410m \
    --gene_dim 1024 \
    --num_layers 12 \
    --proj_dim 768 \
    --proj_hidden_dim 3072 \
    --proj_layers 2 \
    --num_cross_layers 3 \
    --num_heads 12 \
    --igc_weight 1.0 \
    --igm_weight 1.0 \
    --igc_loss_type clip \
    --igc_temperature 0.07 \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --device cuda:1 \
    --num_workers 8 \
    --use_mixed_precision \
    --seed $SEED \
    --pretrained_checkpoint $PRETRAINED_CHECKPOINT \
    --output_dir $OUTPUT_DIR \
    --save_every 1 \
    --per_sample