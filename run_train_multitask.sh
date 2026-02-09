#!/bin/bash

# Multi-task ST Pre-training Script with 3 Augmentations (IGC + IIC + IGM)
# Vision: pathoduet (768) + Gene: C2S pythia_410m (1024, 12 layers only)

# Activate conda environment
source /NAS_Storage3/yeomtl/anaconda3/etc/profile.d/conda.sh
conda activate st_conma

# Set tokens
export HUGGINGFACE_TOKEN="Insert your huggingface token"
export WANDB_API_KEY="Insert your wandbapikey"

# Training configuration
TRAIN_IMAGE_DIR="./pt_dataset/st_images"
TRAIN_GENE_FILE="./pt_dataset/st_sentences/top100_sentences.csv"

OUTPUT_DIR="./checkpoints/st_conma_pythia410m_12layers_3aug_clip"
BATCH_SIZE=32
GRAD_ACCUM=4
EPOCHS=12
NUM_GPUS=4
SEED=42

echo "=================================="
echo "Multi-task Training Configuration"
echo "=================================="
echo "Vision Model: pathoduet (768 dim)"
echo "Gene Model: C2S pythia_410m (1024 dim, 12 layers only)"
echo "Max Sequence Length: 512"
echo "Projection Dim: 768"
echo "Augmentation Strategy (ST-ConMa style):"
echo "  - unaug + gene → IGC (CLIP, temp=0.07)"
echo "  - aug1 + aug2 → IIC (CLIP, temp=0.07)"
echo "Multi-task:"
echo "  - IGC (Image-Gene Contrastive): 1.0"
echo "  - IIC (Image-Image Contrastive): 1.0"
echo "  - IGM (Image-Gene Matching): 1.0"
echo "GPUs: $NUM_GPUS"
echo "Effective Batch Size: $((BATCH_SIZE * GRAD_ACCUM * NUM_GPUS))"
echo "=================================="

# Run training with DDP
torchrun --nproc_per_node=$NUM_GPUS train_multitask.py \
    --train_image_dir $TRAIN_IMAGE_DIR \
    --train_gene_file $TRAIN_GENE_FILE \
    --use_augmentation \
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
    --iic_weight 1.0 \
    --igm_weight 1.0 \
    --igc_loss_type clip \
    --iic_loss_type clip \
    --igc_temperature 0.07 \
    --iic_temperature 0.07 \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --except_ft_samples \
    --epochs $EPOCHS \
    --optimizer adamw \
    --lr 1e-4 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --scheduler linear_warmup_cosine \
    --warmup_epochs 1 \
    --min_lr 1e-6 \
    --num_workers 8 \
    --use_mixed_precision \
    --seed $SEED \
    --use_ddp \
    --output_dir $OUTPUT_DIR \
    --save_every 12 \
    --use_wandb \
    --wandb_project ST-ConMa \
    --wandb_run_name "pathoduet-pythia410m-12layers-3aug-clip"
