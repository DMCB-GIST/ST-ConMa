import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, Union, List
import numpy as np
import sys
import os

try:
    from .gather import gather_embeddings_for_contrastive
    from .loss import CLIPLoss #, SigLIPLoss, SimCLRLoss
except ImportError:
    # Fall back to absolute imports when used as a script
    from gather import gather_embeddings_for_contrastive
    from loss import CLIPLoss, SigLIPLoss, SimCLRLoss


class ProjectionHead(nn.Module):
    """
    Projection head to map embeddings to shared multimodal space.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        use_batch_norm: bool = True,
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: input embedding dimension
            hidden_dim: hidden layer dimension
            output_dim: output embedding dimension (shared space)
            num_layers: number of MLP layers (minimum 2)
            use_batch_norm: whether to use batch normalization
            dropout: dropout rate
        """
        super(ProjectionHead, self).__init__()

        assert num_layers >= 2, "num_layers must be at least 2"

        layers = []

        # First layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.projection = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input embeddings [batch_size, input_dim]

        Returns:
            projected embeddings [batch_size, output_dim]
        """
        return self.projection(x)


class ST_AlignmentModel(nn.Module):
    """
    Multimodal model for aligning image and gene representations.

    Architecture:
        Image Encoder → Vision Projection → Shared Space
        Gene Encoder → Gene Projection → Shared Space
        Cross-Attention Fusion: Gene ← Image (for IGM)

    Multi-task Losses:
        - IGC: Image-Gene Contrastive (CLIP-style bidirectional)
        - IIC: Image-Image Contrastive (SSL with augmentations)
        - IGM: Image-Gene Matching (binary classification with cross-attention)
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        gene_encoder: nn.Module,
        tokenizer: Any,
        vision_dim: int,
        gene_dim: int,
        proj_dim: int = 768,
        proj_hidden_dim: int = 3072,
        proj_layers: int = 2,
        num_cross_layers: int = 3,
        num_heads: int = 12,
        freeze_encoders: bool = False,
        loss_type: str = 'clip',
        temperature: float = 0.07,
        learnable_temperature: bool = True,
        use_ring_chunked: bool = False,
        igc_weight: float = 1.0,
        iic_weight: float = 1.0,
        igm_weight: float = 1.0,
        igc_loss_type: Optional[str] = None,
        iic_loss_type: Optional[str] = None,
        igc_temperature: Optional[float] = None,
        iic_temperature: Optional[float] = None,
        device: str = 'cuda',
        layer_index: Optional[int] = None
    ):
        """
        Args:
            vision_encoder: pre-trained vision encoder (e.g., PathoDuet)
            gene_encoder: pre-trained gene encoder (e.g., C2S pythia_410m)
            tokenizer: tokenizer for gene encoder
            vision_dim: output dimension of vision encoder
            gene_dim: output dimension of gene encoder
            proj_dim: dimension of shared embedding space
            proj_hidden_dim: hidden dimension of projection heads
            proj_layers: number of layers in projection heads
            num_cross_layers: number of cross-attention layers for IGM
            num_heads: number of attention heads in cross-attention
            freeze_encoders: whether to freeze encoder weights
            loss_type: type of contrastive loss ('clip' or 'siglip') - used as default for igc_loss_type and iic_loss_type
            temperature: initial temperature for contrastive loss - used as default for igc_temperature and iic_temperature
            learnable_temperature: whether temperature is learnable
            use_ring_chunked: for SigLIP, use ring-based collective permute
            igc_weight: weight for Image-Gene Contrastive loss
            iic_weight: weight for Image-Image Contrastive loss
            igm_weight: weight for Image-Gene Matching loss
            igc_loss_type: type of loss for IGC ('clip' or 'siglip', defaults to loss_type)
            iic_loss_type: type of loss for IIC ('clip' or 'siglip', defaults to loss_type)
            igc_temperature: temperature for IGC loss (defaults to temperature)
            iic_temperature: temperature for IIC loss (defaults to temperature)
            device: device to use for model
            layer_index: which layer to extract from C2S model (None for last layer)
        """
        super(ST_AlignmentModel, self).__init__()

        self.vision_encoder = vision_encoder
        self.gene_encoder = gene_encoder
        self.tokenizer = tokenizer
        self.proj_dim = proj_dim
        self.vision_dim = vision_dim
        self.gene_dim = gene_dim
        self.loss_type = loss_type
        self.use_ring_chunked = use_ring_chunked
        self.device = device
        self.igc_weight = igc_weight
        self.iic_weight = iic_weight
        self.igm_weight = igm_weight
        self.layer_index = layer_index

        # Set individual loss types and temperatures (with fallback to default)
        self.igc_loss_type = igc_loss_type if igc_loss_type is not None else loss_type
        self.iic_loss_type = iic_loss_type if iic_loss_type is not None else loss_type
        self.igc_temperature = igc_temperature if igc_temperature is not None else temperature
        self.iic_temperature = iic_temperature if iic_temperature is not None else temperature

        # Freeze encoders if specified
        if freeze_encoders:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            for param in self.gene_encoder.parameters():
                param.requires_grad = False

        # Projection heads for IGC (Image-Gene Contrastive)
        self.vision_projection = ProjectionHead(
            input_dim=vision_dim,
            hidden_dim=proj_hidden_dim,
            output_dim=proj_dim,
            num_layers=proj_layers,
            dropout=0.1
        )

        self.gene_projection = ProjectionHead(
            input_dim=gene_dim,
            hidden_dim=proj_hidden_dim,
            output_dim=proj_dim,
            num_layers=proj_layers,
            dropout=0.1
        )

        # Projection head for IIC (Image-Image Contrastive) - stronger MLP
        self.img_ssl_projection = ProjectionHead(
            input_dim=vision_dim,
            hidden_dim=proj_hidden_dim,
            output_dim=proj_dim,
            num_layers=3,
            dropout=0.1
        )

        # Dimension alignment for gene to vision (for cross-attention)
        if gene_dim != vision_dim:
            self.gene_dim_proj = nn.Linear(gene_dim, vision_dim)
        else:
            self.gene_dim_proj = nn.Identity()

        # Cross-attention fusion for IGM (Gene ← Image)
        self.cross_attention = MultimodalCrossAttention(
            embed_dim=vision_dim,
            num_heads=num_heads,
            num_layers=num_cross_layers,
            dropout=0.1,
            activation="gelu"
        )

        # IGM head (Image-Gene Matching) - uses fused features
        self.igm_head = nn.Sequential(
            nn.Linear(vision_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(proj_hidden_dim, 1)
        )

        # Loss functions - separate configuration for IGC and IIC
        # IGC (Image-Gene Contrastive) Loss
        if self.igc_loss_type == 'clip':
            self.igc_loss_fn = CLIPLoss(
                temperature=self.igc_temperature,
                learnable_temperature=learnable_temperature
            )
        elif self.igc_loss_type == 'siglip':
            self.igc_loss_fn = SigLIPLoss(
                temperature=self.igc_temperature,
                bias=0.0,
                learnable_temperature=learnable_temperature,
                learnable_bias=False,
                use_ring_chunked=use_ring_chunked
            )
        else:
            raise ValueError(f"Unknown igc_loss_type: {self.igc_loss_type}. Choose 'clip' or 'siglip'.")

        # IIC (Image-Image Contrastive) Loss
        if self.iic_loss_type == 'clip':
            self.iic_loss_fn = CLIPLoss(
                temperature=self.iic_temperature,
                learnable_temperature=learnable_temperature
            )
        elif self.iic_loss_type == 'siglip':
            self.iic_loss_fn = SigLIPLoss(
                temperature=self.iic_temperature,
                bias=0.0,
                learnable_temperature=learnable_temperature,
                learnable_bias=False,
                use_ring_chunked=use_ring_chunked
            )
        elif self.iic_loss_type == 'simclr' or self.iic_loss_type == 'ntxent':
            self.iic_loss_fn = SimCLRLoss(
                temperature=self.iic_temperature,
                learnable_temperature=learnable_temperature
            )
        else:
            raise ValueError(f"Unknown iic_loss_type: {self.iic_loss_type}. Choose 'clip', 'siglip', or 'simclr'.")

        # IGM (Image-Gene Matching) Loss
        self.igm_loss_fn = nn.BCEWithLogitsLoss()

    def encode_vision(
        self,
        images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode images to get both pooled and sequence features.

        Args:
            images: input images [batch_size, C, H, W]

        Returns:
            vision_cls: [CLS] token or pooled features [batch_size, vision_dim]
            vision_patches: patch embeddings [batch_size, num_patches, vision_dim]
        """
        # Get vision features
        vision_patches, vision_representation = self.vision_encoder(images)

        return vision_representation, vision_patches

    def encode_gene(
        self,
        gene_sentences: List[str],
        max_length: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode gene sentences to get both pooled and sequence features using C2S model.

        Args:
            gene_sentences: list of space-separated gene sentences [batch_size]
            max_length: optional max length

        Returns:
            gene_cls: pooled gene features [batch_size, gene_dim]
            gene_seq: gene sentence embeddings [batch_size, seq_len, gene_dim]
            gene_attn_mask: attention mask [batch_size, seq_len]
        """
        # Use tokenizer
        if max_length is None:
            max_length = 512

        # Tokenize
        encoded = self.tokenizer(
            gene_sentences,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=max_length
        )

        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)

        # Get hidden states from C2S model
        outputs = self.gene_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # Select layer based on layer_index
        if self.layer_index is None:
            gene_seq = outputs.hidden_states[-1]  # [batch_size, seq_len, gene_dim] - last layer
        else:
            gene_seq = outputs.hidden_states[self.layer_index]  # [batch_size, seq_len, gene_dim] - specific layer

        # Mean pooling for pooled representation (gene_cls)
        mask = attention_mask.unsqueeze(-1).float() 
        masked_embeddings = gene_seq * mask
        sum_embeddings = masked_embeddings.sum(dim=1)  # [batch_size, gene_dim]
        sum_mask = mask.sum(dim=1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        gene_cls = sum_embeddings / sum_mask  # [batch_size, gene_dim]

        return gene_cls, gene_seq, attention_mask

    @torch.no_grad()
    def encode_vision_inference(
        self,
        images: torch.Tensor,
        return_numpy: bool = True
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Encode images during inference (no gradients).

        Args:
            images: input images [batch_size, C, H, W]
            return_numpy: whether to return numpy array

        Returns:
            vision projections [batch_size, proj_dim]
        """
        vision_cls, _ = self.encode_vision(images)
        vision_projs = self.vision_projection(vision_cls)
        vision_projs = F.normalize(vision_projs, dim=-1)

        if return_numpy:
            return vision_projs.detach().cpu().numpy()
        else:
            return vision_projs

    @torch.no_grad()
    def encode_gene_inference(
        self,
        gene_sentences: List[str],
        max_length: Optional[int] = None,
        return_numpy: bool = True
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Encode gene sentences during inference (no gradients).

        Args:
            gene_sentences: list of space-separated gene sentences [batch_size]
            max_length: optional maximum sequence length
            return_numpy: whether to return numpy array

        Returns:
            gene projections [batch_size, proj_dim]
        """
        gene_cls, _, _ = self.encode_gene(gene_sentences, max_length=max_length)
        gene_projs = self.gene_projection(gene_cls)
        gene_projs = F.normalize(gene_projs, dim=-1)

        if return_numpy:
            return gene_projs.detach().cpu().numpy()
        else:
            return gene_projs

    def forward(
        self,
        images: torch.Tensor,
        gene_sentences: List[str],
        images_aug1: Optional[torch.Tensor] = None,
        images_aug2: Optional[torch.Tensor] = None,
        images_aug3: Optional[torch.Tensor] = None,
        images_aug: Optional[torch.Tensor] = None,  # Backward compatibility
        max_length: Optional[int] = None,
        return_loss: bool = True,
        use_distributed_gather: bool = True
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with multi-task losses (IGC, IIC, IGM).

        Args:
            images: input images [batch_size, C, H, W] (unused, kept for compatibility)
            gene_sentences: list of space-separated gene sentences [batch_size]
            images_aug1: unaugmented images for IGC [batch_size, C, H, W]
            images_aug2: augmented images for IIC [batch_size, C, H, W]
            images_aug3: augmented images for IIC [batch_size, C, H, W]
            images_aug: deprecated, for backward compatibility
            max_length: optional maximum sequence length for genes
            return_loss: whether to return loss or full outputs
            use_distributed_gather: whether to gather from all GPUs (DDP)

        Returns:
            loss tensor or dict with losses and embeddings

        Note:
            - IGC: images_aug1 (weak aug) vs gene_sentences
            - IIC: images_aug2 (strong aug) vs images_aug3 (strong aug)
            - IGM: creates negative pairs internally by shuffling gene sentences
        """
        # ======================================================================
        # Encode modalities (unified encoding)
        # ======================================================================
        # Backward compatibility: if images_aug is provided, use it for all augmentations
        if images_aug1 is None and images_aug is not None:
            images_aug1 = images_aug
            images_aug2 = images_aug
            images_aug3 = images_aug

        # Use aug1 for IGC (weak augmentation with gene)
        vision_representation, vision_patches = self.encode_vision(images_aug1)
        gene_cls, gene_seq, gene_attn_mask = self.encode_gene(gene_sentences, max_length=max_length)

        # Project to shared space for contrastive learning
        vision_projs = self.vision_projection(vision_representation)
        gene_projs = self.gene_projection(gene_cls)

        outputs = {
            'vision_projs': vision_projs,
            'gene_projs': gene_projs,
            'vision_cls': vision_representation,
            'vision_patches': vision_patches,
            'gene_cls': gene_cls,
            'gene_seq': gene_seq,
            'gene_attn_mask': gene_attn_mask
        }

        if not return_loss:
            return outputs

        # Initialize total loss and loss dict
        total_loss = 0.0
        loss_dict = {}

        # ======================================================================
        # 1. IGC Loss: Image-Gene Contrastive (CLIP-style)
        # ======================================================================
        if self.igc_weight > 0:
            if self.igc_loss_type == 'siglip' and self.use_ring_chunked:
                igc_loss = self.igc_loss_fn(vision_projs, gene_projs)
            else:
                if use_distributed_gather:
                    try:
                        vision_projs_all, gene_projs_all, _, _ = \
                            gather_embeddings_for_contrastive(vision_projs, gene_projs)
                    except:
                        vision_projs_all = vision_projs
                        gene_projs_all = gene_projs
                else:
                    vision_projs_all = vision_projs
                    gene_projs_all = gene_projs

                igc_loss = self.igc_loss_fn(vision_projs_all, gene_projs_all)

            loss_dict['igc_loss'] = igc_loss
            total_loss += self.igc_weight * igc_loss

        # ======================================================================
        # 2. IIC Loss: Image-Image Contrastive (SSL with augmentations)
        # ======================================================================
        if self.iic_weight > 0 and images_aug2 is not None and images_aug3 is not None:
            # Encode aug2 and aug3 (strong augmentations)
            vision_representation_aug2, _ = self.encode_vision(images_aug2)
            vision_representation_aug3, _ = self.encode_vision(images_aug3)

            # Project through SSL head
            img_ssl_projs_aug2 = self.img_ssl_projection(vision_representation_aug2)
            img_ssl_projs_aug3 = self.img_ssl_projection(vision_representation_aug3)

            if self.iic_loss_type == 'siglip' and self.use_ring_chunked:
                iic_loss = self.iic_loss_fn(img_ssl_projs_aug2, img_ssl_projs_aug3)
            else:
                if use_distributed_gather:
                    try:
                        img_ssl_projs_aug2_all, img_ssl_projs_aug3_all, _, _ = \
                            gather_embeddings_for_contrastive(img_ssl_projs_aug2, img_ssl_projs_aug3)
                    except:
                        img_ssl_projs_aug2_all = img_ssl_projs_aug2
                        img_ssl_projs_aug3_all = img_ssl_projs_aug3
                else:
                    img_ssl_projs_aug2_all = img_ssl_projs_aug2
                    img_ssl_projs_aug3_all = img_ssl_projs_aug3

                iic_loss = self.iic_loss_fn(img_ssl_projs_aug2_all, img_ssl_projs_aug3_all)

            loss_dict['iic_loss'] = iic_loss
            total_loss += self.iic_weight * iic_loss

        # ======================================================================
        # 3. IGM Loss: Image-Gene Matching with Cross-Attention
        # ======================================================================
        if self.igm_weight > 0:
            batch_size = gene_seq.shape[0]

            # Create negative pairs by shuffling gene sentences
            # Ensure no sample is paired with itself
            perm = torch.randperm(batch_size, device=gene_seq.device)

            # Fix any indices that point to themselves
            for i in range(batch_size):
                if perm[i] == i:
                    # Swap with next index (wrap around if needed)
                    j = (i + 1) % batch_size
                    perm[i], perm[j] = perm[j], perm[i]

            gene_seq_negative = gene_seq[perm]
            gene_attn_mask_negative = gene_attn_mask[perm]

            # Concatenate positive and negative pairs: [batch_size * 2, seq_len, dim]
            gene_seq_combined = torch.cat([gene_seq, gene_seq_negative], dim=0)
            gene_attn_mask_combined = torch.cat([gene_attn_mask, gene_attn_mask_negative], dim=0)

            # Duplicate vision patches for matching: [batch_size * 2, num_patches, dim]
            vision_patches_combined = torch.cat([vision_patches, vision_patches], dim=0)

            # Create labels: [1,1,1..., 0,0,0...]
            igm_labels = torch.cat([
                torch.ones(batch_size, device=gene_seq.device),
                torch.zeros(batch_size, device=gene_seq.device)
            ])

            # Align gene dimension to vision dimension
            gene_seq_aligned = self.gene_dim_proj(gene_seq_combined)

            # Cross-attention fusion: Gene ← Image
            fused_gene = self.cross_attention(
                gene_features=gene_seq_aligned,
                image_features=vision_patches_combined,
                gene_attn_mask=gene_attn_mask_combined,
                image_attn_mask=None
            )

            # Average pooling across sequence (excluding padding)
            # Since gene encoder has no CLS token, use masked average pooling
            mask = gene_attn_mask_combined.unsqueeze(-1).float()  # [batch_size * 2, seq_len, 1]
            masked_fused = fused_gene * mask  # [batch_size * 2, seq_len, vision_dim]
            sum_fused = masked_fused.sum(dim=1)  # [batch_size * 2, vision_dim]
            sum_mask = mask.sum(dim=1)  # [batch_size * 2, 1]
            sum_mask = torch.clamp(sum_mask, min=1e-9)  # Avoid division by zero
            fused_gene_cls = sum_fused / sum_mask  # [batch_size * 2, vision_dim]

            # Predict matching score
            igm_logits = self.igm_head(fused_gene_cls).squeeze(-1)  # [batch_size * 2]

            # Compute BCE loss
            igm_loss = self.igm_loss_fn(igm_logits, igm_labels.float())

            loss_dict['igm_loss'] = igm_loss
            loss_dict['igm_logits'] = igm_logits
            total_loss += self.igm_weight * igm_loss

        loss_dict['total_loss'] = total_loss

        # Return loss_dict (with all losses) when return_loss=True
        # This allows tracking individual losses (igc, iic, igm)
        return loss_dict


def create_vision_gene_model(
    vision_encoder: nn.Module,
    gene_encoder: nn.Module,
    tokenizer: Any,
    vision_dim: int,
    gene_dim: int,
    config: Optional[Dict[str, Any]] = None
) -> ST_AlignmentModel:
    """
    Factory function to create vision-gene alignment model.

    Args:
        vision_encoder: pre-trained vision encoder
        gene_encoder: pre-trained gene encoder
        tokenizer: tokenizer for gene encoder
        vision_dim: vision encoder output dimension
        gene_dim: gene encoder output dimension
        config: optional configuration dict

    Returns:
        ST_AlignmentModel instance

    """
    if config is None:
        config = {}

    default_config = {
        'proj_dim': 768,
        'proj_hidden_dim': 3072,
        'proj_layers': 2,
        'freeze_encoders': False,
        'loss_type': 'clip',
        'temperature': 0.07,
        'learnable_temperature': False,
        'use_ring_chunked': False,
        'device': 'cuda'
    }

    # Merge with default config
    for key, value in default_config.items():
        if key not in config:
            config[key] = value

    model = ST_AlignmentModel(
        vision_encoder=vision_encoder,
        gene_encoder=gene_encoder,
        tokenizer=tokenizer,
        vision_dim=vision_dim,
        gene_dim=gene_dim,
        **config
    )

    return model


class ST_AlignmentTrainer:
    """
    Trainer for vision-gene alignment model with gradient accumulation support.
    """

    def __init__(
        self,
        model: ST_AlignmentModel,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        device: str = 'cuda',
        use_mixed_precision: bool = True,
        gradient_accumulation_steps: int = 4,
        max_grad_norm: Optional[float] = 1.0
    ):
      
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.use_mixed_precision = use_mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm

        if use_mixed_precision:
            self.scaler = torch.amp.GradScaler()

        # Track accumulation step and loss
        self.accumulation_counter = 0
        self.accumulated_loss = 0.0

        # Track individual losses for logging
        self.accumulated_igc_loss = 0.0
        self.accumulated_iic_loss = 0.0
        self.accumulated_igm_loss = 0.0

    def train_step(
        self,
        gene_sentences: List[str],
        images: Optional[torch.Tensor] = None,  # For compatibility, can be None
        images_aug1: Optional[torch.Tensor] = None,
        images_aug2: Optional[torch.Tensor] = None,
        images_aug3: Optional[torch.Tensor] = None,
        images_aug: Optional[torch.Tensor] = None,  # Backward compatibility
        max_length: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Single training step with multi-task losses and gradient accumulation.

        Args:
            gene_sentences: list of space-separated gene sentences [batch_size]
            images: batch of images [batch_size, C, H, W] (optional, defaults to images_aug1)
            max_length: optional maximum sequence length for genes

        Returns:
            dict with losses (only when gradients are updated)
        """
        self.model.train()

        # For compatibility: if images is None, use images_aug1
        # (model.forward expects images but doesn't actually use it)
        if images is None:
            images = images_aug1

        # Move to device
        if images is not None:
            images = images.to(self.device)
        if images_aug1 is not None:
            images_aug1 = images_aug1.to(self.device)
        if images_aug2 is not None:
            images_aug2 = images_aug2.to(self.device)
        if images_aug3 is not None:
            images_aug3 = images_aug3.to(self.device)
        # Backward compatibility
        if images_aug is not None:
            images_aug = images_aug.to(self.device)

        # Forward pass with mixed precision
        if self.use_mixed_precision:
            with torch.amp.autocast(device_type='cuda'):
                loss_dict = self.model(
                    images=images,
                    gene_sentences=gene_sentences,
                    images_aug1=images_aug1,
                    images_aug2=images_aug2,
                    images_aug3=images_aug3,
                    images_aug=images_aug,  # Backward compatibility
                    max_length=max_length,
                    return_loss=True  # Returns loss_dict with total_loss and individual losses
                )

            total_loss = loss_dict['total_loss']

            # Accumulate loss (before scaling for gradient)
            self.accumulated_loss += total_loss.item()

            # Store and accumulate individual losses for logging
            self.current_loss_dict = {k: v.item() if isinstance(v, torch.Tensor) and v.numel() == 1 else v
                                     for k, v in loss_dict.items() if 'loss' in k}

            # Accumulate individual losses
            if 'igc_loss' in loss_dict:
                self.accumulated_igc_loss += loss_dict['igc_loss'].item()
            if 'iic_loss' in loss_dict:
                self.accumulated_iic_loss += loss_dict['iic_loss'].item()
            if 'igm_loss' in loss_dict:
                self.accumulated_igm_loss += loss_dict['igm_loss'].item()

            # Scale loss by accumulation steps
            total_loss = total_loss / self.gradient_accumulation_steps

            # Backward pass
            self.scaler.scale(total_loss).backward()
        else:
            loss_dict = self.model(
                images=images,
                gene_sentences=gene_sentences,
                images_aug1=images_aug1,
                images_aug2=images_aug2,
                images_aug3=images_aug3,
                images_aug=images_aug,  # Backward compatibility
                max_length=max_length,
                return_loss=True  # Returns loss_dict with total_loss and individual losses
            )

            total_loss = loss_dict['total_loss']

            # Accumulate loss (before scaling for gradient)
            self.accumulated_loss += total_loss.item()

            # Store and accumulate individual losses for logging
            self.current_loss_dict = {k: v.item() if isinstance(v, torch.Tensor) and v.numel() == 1 else v
                                     for k, v in loss_dict.items() if 'loss' in k}

            # Accumulate individual losses
            if 'igc_loss' in loss_dict:
                self.accumulated_igc_loss += loss_dict['igc_loss'].item()
            if 'iic_loss' in loss_dict:
                self.accumulated_iic_loss += loss_dict['iic_loss'].item()
            if 'igm_loss' in loss_dict:
                self.accumulated_igm_loss += loss_dict['igm_loss'].item()

            # Scale loss by accumulation steps
            total_loss = total_loss / self.gradient_accumulation_steps

            # Backward pass
            total_loss.backward()

        # Increment accumulation counter
        self.accumulation_counter += 1

        # Update weights if accumulation is complete
        metrics = {}
        if self.accumulation_counter >= self.gradient_accumulation_steps:
            if self.use_mixed_precision:
                # Gradient clipping
                if self.max_grad_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:
                # Gradient clipping
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )

                # Optimizer step
                self.optimizer.step()

            # Zero gradients
            self.optimizer.zero_grad()

            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()

            # Return average loss across accumulation steps
            metrics['total_loss'] = self.accumulated_loss / self.gradient_accumulation_steps
            metrics['updated'] = True

            # Add individual losses
            if self.accumulated_igc_loss > 0:
                metrics['igc_loss'] = self.accumulated_igc_loss / self.gradient_accumulation_steps
            if self.accumulated_iic_loss > 0:
                metrics['iic_loss'] = self.accumulated_iic_loss / self.gradient_accumulation_steps
            if self.accumulated_igm_loss > 0:
                metrics['igm_loss'] = self.accumulated_igm_loss / self.gradient_accumulation_steps

            # Reset accumulation counter and loss
            self.accumulation_counter = 0
            self.accumulated_loss = 0.0
            self.accumulated_igc_loss = 0.0
            self.accumulated_iic_loss = 0.0
            self.accumulated_igm_loss = 0.0
        else:
            # Don't return loss for non-update steps
            metrics['total_loss'] = 0.0
            metrics['updated'] = False

        return metrics

# ============================================================================
# Cross-Attention Based Multimodal Fusion (ST-ConMa style)
# ============================================================================

class CrossAttentionLayer(nn.Module):
    """
    Cross-attention layer for fusing multimodal information.

    Architecture:
        1. Self-attention on query modality
        2. Cross-attention: query attends to key-value modality
        3. Feed-forward network
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 12,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        """
        Args:
            embed_dim: embedding dimension
            num_heads: number of attention heads
            dropout: dropout rate
            activation: activation function ('gelu' or 'relu')
        """
        super(CrossAttentionLayer, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Self-attention on query
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Cross-attention: query → key-value
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        kv_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        # 1. Self-attention on query
        key_padding_mask_q = None
        if query_mask is not None:
            key_padding_mask_q = (query_mask == 0)  # True for padding positions

        residual = query
        query = self.norm1(query)
        query2, _ = self.self_attn(
            query=query,
            key=query,
            value=query,
            key_padding_mask=key_padding_mask_q,
            need_weights=False
        )
        query = residual + self.dropout1(query2)

        # 2. Cross-attention: query → key-value
        key_padding_mask_kv = None
        if kv_mask is not None:
            key_padding_mask_kv = (kv_mask == 0)

        residual = query
        query = self.norm2(query)

        # Apply mask to query before cross-attention
        if query_mask is not None:
            query = query * query_mask.unsqueeze(-1)

        query2, _ = self.cross_attn(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=key_padding_mask_kv,
            need_weights=False
        )

        # Apply mask to cross-attention output
        if query_mask is not None:
            query2 = query2 * query_mask.unsqueeze(-1)

        query = residual + self.dropout2(query2)

        # 3. Feed-forward network
        residual = query
        query = self.norm3(query)
        query2 = self.ffn(query)
        query = residual + self.dropout3(query2)

        return query


class MultimodalCrossAttention(nn.Module):
    """
    Multi-layer cross-attention for multimodal fusion.

    Fuses gene features with image features using stacked cross-attention layers.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 12,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        """
        Args:
            embed_dim: embedding dimension
            num_heads: number of attention heads
            num_layers: number of cross-attention layers
            dropout: dropout rate
            activation: activation function
        """
        super(MultimodalCrossAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Stack of cross-attention layers
        self.cross_layers = nn.ModuleList([
            CrossAttentionLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                activation=activation
            ) for _ in range(num_layers)
        ])

    def forward(
        self,
        gene_features: torch.Tensor,
        image_features: torch.Tensor,
        gene_attn_mask: Optional[torch.Tensor] = None,
        image_attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Fuse gene and image features via cross-attention

        Args:
            gene_features: gene embeddings [batch_size, gene_seq_len, embed_dim]
            image_features: image embeddings [batch_size, img_seq_len, embed_dim]
            gene_attn_mask: gene attention mask [batch_size, gene_seq_len]
            image_attn_mask: image attention mask [batch_size, img_seq_len]

        Returns:
            fused gene features [batch_size, gene_seq_len, embed_dim]
        """
        batch_size = gene_features.shape[0]
        gene_seq_len = gene_features.shape[1]
        img_seq_len = image_features.shape[1]

        # Create default masks if not provided
        if gene_attn_mask is None:
            gene_attn_mask = torch.ones(
                batch_size, gene_seq_len,
                device=gene_features.device
            )

        if image_attn_mask is None:
            image_attn_mask = torch.ones(
                batch_size, img_seq_len,
                device=image_features.device
            )

        # Apply cross-attention layers
        fused_features = gene_features
        for layer in self.cross_layers:
            fused_features = layer(
                query=fused_features,
                key_value=image_features,
                query_mask=gene_attn_mask,
                kv_mask=image_attn_mask
            )

        return fused_features


