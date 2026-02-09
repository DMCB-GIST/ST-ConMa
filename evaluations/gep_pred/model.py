import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple, Dict

# Add paths for model loading
sys.path.append("./PathoDuet")

from vits import VisionTransformerMoCo


# ============================================================================
# PathoDuet Wrapper
# ============================================================================

class PathoDuetWrapper(nn.Module):
    """Wrapper for PathoDuet model to extract features."""

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        """
        Args:
            x: input images [batch_size, 3, H, W]
        Returns:
            patch_features: [batch_size, num_patches, 768]
            representation: [batch_size, 768]
        """
        patch_features, representation = self.base_model(x)
        patch_features = patch_features[:, 2:, :]  # Remove pretext tokens
        return patch_features, representation


# ============================================================================
# Projection Head (matching pretraining)
# ============================================================================

class ProjectionHead(nn.Module):
    """
    Projection head matching pretraining configuration.
    2-layer MLP: Linear → BatchNorm → GELU → Dropout → Linear
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        layers = []

        # First layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # Hidden layers (if num_layers > 2)
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.projection = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


# ============================================================================
# CLIP Loss (matching pretraining)
# ============================================================================

class CLIPLoss(nn.Module):

    def __init__(self, temperature=0.07, learnable_temperature=True):
        super().__init__()

        if learnable_temperature:
            # Store as logit_scale in log space (following OpenAI CLIP)
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1.0 / temperature))
        else:
            self.register_buffer('logit_scale', torch.tensor(np.log(1.0 / temperature)))

    def forward(self, modality1_features, modality2_features):

        # Normalize features
        modality1_features = F.normalize(modality1_features, dim=-1)
        modality2_features = F.normalize(modality2_features, dim=-1)

        # Get logit scale (equivalent to 1/temperature)
        logit_scale = self.logit_scale.exp()

        # Compute similarity matrix
        logits_per_modality1 = torch.matmul(modality1_features, modality2_features.T) * logit_scale
        logits_per_modality2 = logits_per_modality1.T

        # Ground truth: diagonal elements are positive pairs
        batch_size = modality1_features.shape[0]
        labels = torch.arange(batch_size, device=modality1_features.device)

        # Compute cross-entropy loss in both directions
        loss_m1_to_m2 = F.cross_entropy(logits_per_modality1, labels)
        loss_m2_to_m1 = F.cross_entropy(logits_per_modality2, labels)

        # Average loss from both directions
        loss = (loss_m1_to_m2 + loss_m2_to_m1) / 2

        return loss


class STConMaFinetune(nn.Module):

    def __init__(
        self,
        pretrained_checkpoint: str = None,
        vision_dim: int = 768,
        gene_dim: int = 1024,
        proj_dim: int = 768,
        proj_hidden_dim: int = 3072,
        proj_layers: int = 2,
        num_gene_layers: int = 12,
        max_seq_len: int = 512,
        temperature: float = 0.07,
        learnable_temperature: bool = True,
        freeze_vision_encoder: bool = False,
        freeze_gene_encoder: bool = False,
        dropout: float = 0.1,
        igc_weight: float = 1.0,
        igm_weight: float = 0.0,  # Kept for backward compatibility, not used
        device: str = 'cuda'
    ):
        super().__init__()

        self.vision_dim = vision_dim
        self.gene_dim = gene_dim
        self.proj_dim = proj_dim
        self.max_seq_len = max_seq_len
        self.device = device
        self.num_gene_layers = num_gene_layers
        self.igc_weight = igc_weight

        # ====================================================================
        # Load Vision Encoder (PathoDuet)
        # ====================================================================
        print("Loading PathoDuet vision encoder...")
        base_model = VisionTransformerMoCo(pretext_token=True, global_pool="avg")

        # Load original PathoDuet checkpoint
        pathoduet_ckpt = torch.load(
            "./PathoDuet/checkpoint/checkpoint_HE.pth",
            map_location=device
        )
        base_model.load_state_dict(pathoduet_ckpt, strict=False)
        base_model.head = nn.Identity()

        self.vision_encoder = PathoDuetWrapper(base_model)

        # ====================================================================
        # Load Gene Encoder (C2S pythia_410m)
        # ====================================================================
        print("Loading C2S gene encoder (pythia_410m)...")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = "vandijklab/C2S-Pythia-410m-diverse-single-and-multi-cell-tasks"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.gene_encoder = AutoModelForCausalLM.from_pretrained(model_id)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Truncate to num_gene_layers
        original_layers = len(self.gene_encoder.gpt_neox.layers)
        if num_gene_layers < original_layers:
            self.gene_encoder.gpt_neox.layers = self.gene_encoder.gpt_neox.layers[:num_gene_layers]
            print(f"Truncated gene encoder from {original_layers} to {num_gene_layers} layers")

        # ====================================================================
        # Projection Heads for IGC (matching pretraining)
        # ====================================================================
        self.vision_projection = ProjectionHead(
            input_dim=vision_dim,
            hidden_dim=proj_hidden_dim,
            output_dim=proj_dim,
            num_layers=proj_layers,
            dropout=dropout
        )

        self.gene_projection = ProjectionHead(
            input_dim=gene_dim,
            hidden_dim=proj_hidden_dim,
            output_dim=proj_dim,
            num_layers=proj_layers,
            dropout=dropout
        )

        # ====================================================================
        # IGC Loss (CLIP-style, matching pretraining)
        # ====================================================================
        self.igc_loss_fn = CLIPLoss(
            temperature=temperature,
            learnable_temperature=learnable_temperature
        )

        # ====================================================================
        # Load Pretrained Checkpoint
        # ====================================================================
        if pretrained_checkpoint is not None and pretrained_checkpoint != "" and pretrained_checkpoint.lower() != "none":
            print(f"Loading pretrained checkpoint: {pretrained_checkpoint}")
            self._load_checkpoint(pretrained_checkpoint)

        # ====================================================================
        # Freeze Encoders if specified
        # ====================================================================
        if freeze_vision_encoder:
            print("Freezing vision encoder...")
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

        if freeze_gene_encoder:
            print("Freezing gene encoder...")
            for param in self.gene_encoder.parameters():
                param.requires_grad = False

    def _load_checkpoint(self, checkpoint_path: str):
        """Load pretrained weights from ST-ConMa checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Get state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Remove 'module.' prefix if exists (from DDP)
        if any(key.startswith('module.') for key in state_dict.keys()):
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace('module.', '')
                new_state_dict[new_key] = value
            state_dict = new_state_dict

        # Load vision_encoder weights
        vision_state = {k.replace('vision_encoder.', ''): v
                       for k, v in state_dict.items()
                       if k.startswith('vision_encoder.')}
        if vision_state:
            self.vision_encoder.load_state_dict(vision_state, strict=False)
            print(f"  Vision encoder: loaded {len(vision_state)} weights")

        # Load gene_encoder weights
        gene_state = {k.replace('gene_encoder.', ''): v
                     for k, v in state_dict.items()
                     if k.startswith('gene_encoder.')}
        if gene_state:
            self.gene_encoder.load_state_dict(gene_state, strict=False)
            print(f"  Gene encoder: loaded {len(gene_state)} weights")

        # Load vision_projection weights
        vision_proj_state = {k.replace('vision_projection.', ''): v
                           for k, v in state_dict.items()
                           if k.startswith('vision_projection.')}
        if vision_proj_state:
            self.vision_projection.load_state_dict(vision_proj_state, strict=False)
            print(f"  Vision projection: loaded {len(vision_proj_state)} weights")

        # Load gene_projection weights
        gene_proj_state = {k.replace('gene_projection.', ''): v
                         for k, v in state_dict.items()
                         if k.startswith('gene_projection.')}
        if gene_proj_state:
            self.gene_projection.load_state_dict(gene_proj_state, strict=False)
            print(f"  Gene projection: loaded {len(gene_proj_state)} weights")

        # Load IGC loss weights (logit_scale)
        igc_loss_state = {k.replace('igc_loss_fn.', ''): v
                        for k, v in state_dict.items()
                        if k.startswith('igc_loss_fn.')}
        if igc_loss_state:
            self.igc_loss_fn.load_state_dict(igc_loss_state, strict=False)
            print(f"  IGC loss: loaded {len(igc_loss_state)} weights")

        print("Checkpoint loaded successfully!")

    def encode_vision(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode images.
        Returns:
            representation: [B, vision_dim] pooled features
            patches: [B, num_patches, vision_dim] patch features
        """
        patches, representation = self.vision_encoder(images)
        return representation, patches

    def encode_gene(
        self,
        gene_sentences: List[str],
        max_length: Optional[int] = None
    ) -> torch.Tensor:
        """
        Encode gene sentences using C2S model.

        Args:
            gene_sentences: list of space-separated gene names
            max_length: optional max length

        Returns:
            gene_cls: [B, gene_dim] mean-pooled gene embeddings
        """
        if max_length is None:
            max_length = self.max_seq_len

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

        # Get hidden states
        outputs = self.gene_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # Use last hidden state
        gene_seq = outputs.hidden_states[-1]  # [B, seq_len, gene_dim]

        # Mean pooling (exclude padding)
        mask = attention_mask.unsqueeze(-1).float()
        masked_hidden = gene_seq * mask
        sum_hidden = masked_hidden.sum(dim=1)
        sum_mask = mask.sum(dim=1).clamp(min=1e-9)
        gene_cls = sum_hidden / sum_mask  # [B, gene_dim]

        return gene_cls

    def forward(
        self,
        images: torch.Tensor,
        gene_sentences: List[str],
        max_length: Optional[int] = None,
        return_embeddings: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with IGC loss.

        Args:
            images: [B, 3, H, W] input images
            gene_sentences: list of space-separated gene names
            max_length: optional max sequence length
            return_embeddings: whether to return embeddings

        Returns:
            dict with 'loss', 'igc_loss' and optionally embeddings
        """
        # Encode vision
        vision_repr, _ = self.encode_vision(images)

        # Encode gene
        gene_cls = self.encode_gene(gene_sentences, max_length)

        # Project to shared space
        vision_projs = self.vision_projection(vision_repr)
        gene_projs = self.gene_projection(gene_cls)

        # IGC Loss: Image-Gene Contrastive (CLIP-style)
        igc_loss = self.igc_loss_fn(vision_projs, gene_projs)
        loss = self.igc_weight * igc_loss

        result = {
            'loss': loss,
            'total_loss': loss,
            'igc_loss': igc_loss,
            'igm_loss': torch.tensor(0.0, device=images.device)  # For compatibility
        }

        if return_embeddings:
            result['vision_projs'] = vision_projs
            result['gene_projs'] = gene_projs
            result['image_embeddings'] = vision_projs
            result['gene_embeddings'] = gene_projs

        return result

    @torch.no_grad()
    def get_image_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        """Get projected image embeddings for inference."""
        vision_repr, _ = self.encode_vision(images)
        image_embeddings = self.vision_projection(vision_repr)
        image_embeddings = F.normalize(image_embeddings, dim=-1)
        return image_embeddings

    @torch.no_grad()
    def get_gene_embeddings(self, gene_sentences: List[str]) -> torch.Tensor:
        """Get projected gene embeddings for inference."""
        gene_cls = self.encode_gene(gene_sentences)
        gene_embeddings = self.gene_projection(gene_cls)
        gene_embeddings = F.normalize(gene_embeddings, dim=-1)
        return gene_embeddings


# ============================================================================
# Cross-Attention Layers for IGM
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
        super().__init__()
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
            key_padding_mask_q = (query_mask == 0)

        residual = query
        query = self.norm1(query)
        query2, _ = self.self_attn(
            query=query, key=query, value=query,
            key_padding_mask=key_padding_mask_q, need_weights=False
        )
        query = residual + self.dropout1(query2)

        # 2. Cross-attention: query → key-value
        key_padding_mask_kv = None
        if kv_mask is not None:
            key_padding_mask_kv = (kv_mask == 0)

        residual = query
        query = self.norm2(query)
        if query_mask is not None:
            query = query * query_mask.unsqueeze(-1)

        query2, _ = self.cross_attn(
            query=query, key=key_value, value=key_value,
            key_padding_mask=key_padding_mask_kv, need_weights=False
        )

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
    """Multi-layer cross-attention for multimodal fusion."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 12,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

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
        batch_size = gene_features.shape[0]
        gene_seq_len = gene_features.shape[1]
        img_seq_len = image_features.shape[1]

        if gene_attn_mask is None:
            gene_attn_mask = torch.ones(batch_size, gene_seq_len, device=gene_features.device)
        if image_attn_mask is None:
            image_attn_mask = torch.ones(batch_size, img_seq_len, device=image_features.device)

        fused_features = gene_features
        for layer in self.cross_layers:
            fused_features = layer(
                query=fused_features,
                key_value=image_features,
                query_mask=gene_attn_mask,
                kv_mask=image_attn_mask
            )

        return fused_features


# ============================================================================
# ST-ConMa Fine-tuning Model with IGC + IGM (Full)
# ============================================================================

class STConMaFinetuneFull(nn.Module):
    """
    ST-ConMa model for fine-tuning with both IGC and IGM losses.

    Compared to STConMaFinetune:
    - Adds cross-attention layers for gene ← image fusion
    - Adds IGM (Image-Gene Matching) head and loss
    - Uses both IGC and IGM losses during training
    """

    def __init__(
        self,
        pretrained_checkpoint: str = None,
        vision_dim: int = 768,
        gene_dim: int = 1024,
        proj_dim: int = 768,
        proj_hidden_dim: int = 3072,
        proj_layers: int = 2,
        num_gene_layers: int = 12,
        num_cross_layers: int = 3,
        num_heads: int = 12,
        max_seq_len: int = 512,
        temperature: float = 0.07,
        learnable_temperature: bool = True,
        freeze_vision_encoder: bool = False,
        freeze_gene_encoder: bool = False,
        dropout: float = 0.1,
        igc_weight: float = 1.0,
        igm_weight: float = 1.0,
        device: str = 'cuda'
    ):
        super().__init__()

        self.vision_dim = vision_dim
        self.gene_dim = gene_dim
        self.proj_dim = proj_dim
        self.max_seq_len = max_seq_len
        self.device = device
        self.num_gene_layers = num_gene_layers
        self.igc_weight = igc_weight
        self.igm_weight = igm_weight

        # ====================================================================
        # Load Vision Encoder (PathoDuet)
        # ====================================================================
        print("Loading PathoDuet vision encoder...")
        base_model = VisionTransformerMoCo(pretext_token=True, global_pool="avg")

        pathoduet_ckpt = torch.load(
            "./PathoDuet/checkpoint/checkpoint_HE.pth",
            map_location=device
        )
        base_model.load_state_dict(pathoduet_ckpt, strict=False)
        base_model.head = nn.Identity()

        self.vision_encoder = PathoDuetWrapper(base_model)

        # ====================================================================
        # Load Gene Encoder (C2S pythia_410m)
        # ====================================================================
        print("Loading C2S gene encoder (pythia_410m)...")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = "vandijklab/C2S-Pythia-410m-diverse-single-and-multi-cell-tasks"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.gene_encoder = AutoModelForCausalLM.from_pretrained(model_id)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Truncate to num_gene_layers
        original_layers = len(self.gene_encoder.gpt_neox.layers)
        if num_gene_layers < original_layers:
            self.gene_encoder.gpt_neox.layers = self.gene_encoder.gpt_neox.layers[:num_gene_layers]
            print(f"Truncated gene encoder from {original_layers} to {num_gene_layers} layers")

        # ====================================================================
        # Projection Heads for IGC
        # ====================================================================
        self.vision_projection = ProjectionHead(
            input_dim=vision_dim,
            hidden_dim=proj_hidden_dim,
            output_dim=proj_dim,
            num_layers=proj_layers,
            dropout=dropout
        )

        self.gene_projection = ProjectionHead(
            input_dim=gene_dim,
            hidden_dim=proj_hidden_dim,
            output_dim=proj_dim,
            num_layers=proj_layers,
            dropout=dropout
        )

        # ====================================================================
        # Cross-Attention for IGM (Gene ← Image)
        # ====================================================================
        # Dimension alignment for gene to vision
        if gene_dim != vision_dim:
            self.gene_dim_proj = nn.Linear(gene_dim, vision_dim)
        else:
            self.gene_dim_proj = nn.Identity()

        self.cross_attention = MultimodalCrossAttention(
            embed_dim=vision_dim,
            num_heads=num_heads,
            num_layers=num_cross_layers,
            dropout=dropout,
            activation="gelu"
        )

        # IGM head (binary classification)
        self.igm_head = nn.Sequential(
            nn.Linear(vision_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_hidden_dim, 1)
        )

        # ====================================================================
        # Loss Functions
        # ====================================================================
        self.igc_loss_fn = CLIPLoss(
            temperature=temperature,
            learnable_temperature=learnable_temperature
        )
        self.igm_loss_fn = nn.BCEWithLogitsLoss()

        # ====================================================================
        # Load Pretrained Checkpoint
        # ====================================================================
        if pretrained_checkpoint is not None and pretrained_checkpoint != "" and pretrained_checkpoint.lower() != "none":
            print(f"Loading pretrained checkpoint: {pretrained_checkpoint}")
            self._load_checkpoint(pretrained_checkpoint)

        # ====================================================================
        # Freeze Encoders if specified
        # ====================================================================
        if freeze_vision_encoder:
            print("Freezing vision encoder...")
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

        if freeze_gene_encoder:
            print("Freezing gene encoder...")
            for param in self.gene_encoder.parameters():
                param.requires_grad = False

    def _load_checkpoint(self, checkpoint_path: str):
        """Load pretrained weights from ST-ConMa checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Remove 'module.' prefix if exists (from DDP)
        if any(key.startswith('module.') for key in state_dict.keys()):
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace('module.', '')
                new_state_dict[new_key] = value
            state_dict = new_state_dict

        # Load vision_encoder weights
        vision_state = {k.replace('vision_encoder.', ''): v
                       for k, v in state_dict.items()
                       if k.startswith('vision_encoder.')}
        if vision_state:
            self.vision_encoder.load_state_dict(vision_state, strict=False)
            print(f"  Vision encoder: loaded {len(vision_state)} weights")

        # Load gene_encoder weights
        gene_state = {k.replace('gene_encoder.', ''): v
                     for k, v in state_dict.items()
                     if k.startswith('gene_encoder.')}
        if gene_state:
            self.gene_encoder.load_state_dict(gene_state, strict=False)
            print(f"  Gene encoder: loaded {len(gene_state)} weights")

        # Load vision_projection weights
        vision_proj_state = {k.replace('vision_projection.', ''): v
                           for k, v in state_dict.items()
                           if k.startswith('vision_projection.')}
        if vision_proj_state:
            self.vision_projection.load_state_dict(vision_proj_state, strict=False)
            print(f"  Vision projection: loaded {len(vision_proj_state)} weights")

        # Load gene_projection weights
        gene_proj_state = {k.replace('gene_projection.', ''): v
                         for k, v in state_dict.items()
                         if k.startswith('gene_projection.')}
        if gene_proj_state:
            self.gene_projection.load_state_dict(gene_proj_state, strict=False)
            print(f"  Gene projection: loaded {len(gene_proj_state)} weights")

        # Load cross_attention weights (if available)
        cross_attn_state = {k.replace('cross_attention.', ''): v
                          for k, v in state_dict.items()
                          if k.startswith('cross_attention.')}
        if cross_attn_state:
            self.cross_attention.load_state_dict(cross_attn_state, strict=False)
            print(f"  Cross attention: loaded {len(cross_attn_state)} weights")

        # Load gene_dim_proj weights (if available)
        gene_dim_proj_state = {k.replace('gene_dim_proj.', ''): v
                              for k, v in state_dict.items()
                              if k.startswith('gene_dim_proj.')}
        if gene_dim_proj_state and not isinstance(self.gene_dim_proj, nn.Identity):
            self.gene_dim_proj.load_state_dict(gene_dim_proj_state, strict=False)
            print(f"  Gene dim proj: loaded {len(gene_dim_proj_state)} weights")

        # Load igm_head weights (if available)
        igm_head_state = {k.replace('igm_head.', ''): v
                        for k, v in state_dict.items()
                        if k.startswith('igm_head.')}
        if igm_head_state:
            self.igm_head.load_state_dict(igm_head_state, strict=False)
            print(f"  IGM head: loaded {len(igm_head_state)} weights")

        # Load IGC loss weights (logit_scale)
        igc_loss_state = {k.replace('igc_loss_fn.', ''): v
                        for k, v in state_dict.items()
                        if k.startswith('igc_loss_fn.')}
        if igc_loss_state:
            self.igc_loss_fn.load_state_dict(igc_loss_state, strict=False)
            print(f"  IGC loss: loaded {len(igc_loss_state)} weights")

        print("Checkpoint loaded successfully!")

    def encode_vision(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode images. Returns representation and patches."""
        patches, representation = self.vision_encoder(images)
        return representation, patches

    def encode_gene(
        self,
        gene_sentences: List[str],
        max_length: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode gene sentences.
        Returns:
            gene_cls: [B, gene_dim] mean-pooled
            gene_seq: [B, seq_len, gene_dim] sequence embeddings
            attention_mask: [B, seq_len]
        """
        if max_length is None:
            max_length = self.max_seq_len

        encoded = self.tokenizer(
            gene_sentences,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=max_length
        )

        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)

        outputs = self.gene_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        gene_seq = outputs.hidden_states[-1]  # [B, seq_len, gene_dim]

        # Mean pooling
        mask = attention_mask.unsqueeze(-1).float()
        masked_hidden = gene_seq * mask
        sum_hidden = masked_hidden.sum(dim=1)
        sum_mask = mask.sum(dim=1).clamp(min=1e-9)
        gene_cls = sum_hidden / sum_mask  # [B, gene_dim]

        return gene_cls, gene_seq, attention_mask

    def forward(
        self,
        images: torch.Tensor,
        gene_sentences: List[str],
        max_length: Optional[int] = None,
        return_embeddings: bool = True,
        use_igm: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with IGC loss, and optionally IGM loss.

        Args:
            images: [B, 3, H, W] input images
            gene_sentences: list of gene sentences
            max_length: optional max sequence length
            return_embeddings: whether to return embeddings
            use_igm: whether to compute IGM loss (default: True)

        Returns:
            dict with losses and optionally embeddings
        """
        batch_size = images.shape[0]

        # Encode vision
        vision_repr, vision_patches = self.encode_vision(images)

        # Encode gene
        gene_cls, gene_seq, gene_attn_mask = self.encode_gene(gene_sentences, max_length)

        # ================================================================
        # IGC Loss: Image-Gene Contrastive (always computed)
        # ================================================================
        vision_projs = self.vision_projection(vision_repr)
        gene_projs = self.gene_projection(gene_cls)
        igc_loss = self.igc_loss_fn(vision_projs, gene_projs)

        # Initialize result
        result = {
            'igc_loss': igc_loss,
        }

        # ================================================================
        # IGM Loss: Image-Gene Matching with Cross-Attention (optional)
        # ================================================================
        if use_igm and self.igm_weight > 0:
            # Create negative pairs by shuffling gene sequences
            perm = torch.randperm(batch_size, device=gene_seq.device)
            for i in range(batch_size):
                if perm[i] == i:
                    j = (i + 1) % batch_size
                    perm[i], perm[j] = perm[j].clone(), perm[i].clone()

            gene_seq_negative = gene_seq[perm]
            gene_attn_mask_negative = gene_attn_mask[perm]

            # Concatenate positive and negative pairs
            gene_seq_combined = torch.cat([gene_seq, gene_seq_negative], dim=0)
            gene_attn_mask_combined = torch.cat([gene_attn_mask, gene_attn_mask_negative], dim=0)
            vision_patches_combined = torch.cat([vision_patches, vision_patches], dim=0)

            # Labels: [1,1,1..., 0,0,0...]
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

            # Average pooling across sequence
            mask = gene_attn_mask_combined.unsqueeze(-1).float()
            masked_fused = fused_gene * mask
            sum_fused = masked_fused.sum(dim=1)
            sum_mask = mask.sum(dim=1).clamp(min=1e-9)
            fused_gene_cls = sum_fused / sum_mask

            # Predict matching score
            igm_logits = self.igm_head(fused_gene_cls).squeeze(-1)
            igm_loss = self.igm_loss_fn(igm_logits, igm_labels.float())

            # Total loss with IGM
            loss = self.igc_weight * igc_loss + self.igm_weight * igm_loss
            result['igm_loss'] = igm_loss
        else:
            # Total loss without IGM
            loss = self.igc_weight * igc_loss
            result['igm_loss'] = torch.tensor(0.0, device=images.device)

        result['loss'] = loss
        result['total_loss'] = loss

        if return_embeddings:
            result['vision_projs'] = vision_projs
            result['gene_projs'] = gene_projs
            result['image_embeddings'] = vision_projs
            result['gene_embeddings'] = gene_projs

        return result

    @torch.no_grad()
    def get_image_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        """Get projected image embeddings for inference."""
        vision_repr, _ = self.encode_vision(images)
        image_embeddings = self.vision_projection(vision_repr)
        image_embeddings = F.normalize(image_embeddings, dim=-1)
        return image_embeddings

    @torch.no_grad()
    def get_gene_embeddings(self, gene_sentences: List[str]) -> torch.Tensor:
        """Get projected gene embeddings for inference."""
        gene_cls, _, _ = self.encode_gene(gene_sentences)
        gene_embeddings = self.gene_projection(gene_cls)
        gene_embeddings = F.normalize(gene_embeddings, dim=-1)
        return gene_embeddings

    @torch.no_grad()
    def compute_igm_scores(
        self,
        images: torch.Tensor,
        gene_sentences: List[str],
        max_length: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute IGM matching scores between images and gene sentences.

        Args:
            images: [B, 3, H, W] input images
            gene_sentences: list of gene sentences (length B)
            max_length: optional max sequence length

        Returns:
            igm_probs: [B] matching probabilities (sigmoid of logits)
        """
        # Encode vision (get patches for cross-attention)
        _, vision_patches = self.encode_vision(images)

        # Encode gene (get sequence for cross-attention)
        _, gene_seq, gene_attn_mask = self.encode_gene(gene_sentences, max_length)

        # Align gene dimension to vision dimension
        gene_seq_aligned = self.gene_dim_proj(gene_seq)

        # Cross-attention fusion: Gene ← Image
        fused_gene = self.cross_attention(
            gene_features=gene_seq_aligned,
            image_features=vision_patches,
            gene_attn_mask=gene_attn_mask,
            image_attn_mask=None
        )

        # Average pooling across sequence
        mask = gene_attn_mask.unsqueeze(-1).float()
        masked_fused = fused_gene * mask
        sum_fused = masked_fused.sum(dim=1)
        sum_mask = mask.sum(dim=1).clamp(min=1e-9)
        fused_gene_cls = sum_fused / sum_mask

        # Predict matching score and convert to probability
        igm_logits = self.igm_head(fused_gene_cls).squeeze(-1)
        igm_probs = torch.sigmoid(igm_logits)

        return igm_probs

    @torch.no_grad()
    def compute_igm_scores_batch(
        self,
        image: torch.Tensor,
        gene_sentences: List[str],
        gene_seqs: torch.Tensor,
        gene_attn_masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute IGM matching scores for one image against multiple gene candidates.
        More efficient for top-k retrieval scenario.

        Args:
            image: [1, 3, H, W] single image (will be expanded)
            gene_sentences: list of gene sentences (length K, for reference)
            gene_seqs: [K, seq_len, gene_dim] pre-encoded gene sequences
            gene_attn_masks: [K, seq_len] attention masks

        Returns:
            igm_probs: [K] matching probabilities
        """
        K = gene_seqs.shape[0]

        # Encode vision and expand for K candidates
        _, vision_patches = self.encode_vision(image)  # [1, num_patches, vision_dim]
        vision_patches = vision_patches.expand(K, -1, -1)  # [K, num_patches, vision_dim]

        # Align gene dimension to vision dimension
        gene_seq_aligned = self.gene_dim_proj(gene_seqs)  # [K, seq_len, vision_dim]

        # Cross-attention fusion: Gene ← Image
        fused_gene = self.cross_attention(
            gene_features=gene_seq_aligned,
            image_features=vision_patches,
            gene_attn_mask=gene_attn_masks,
            image_attn_mask=None
        )

        # Average pooling across sequence
        mask = gene_attn_masks.unsqueeze(-1).float()
        masked_fused = fused_gene * mask
        sum_fused = masked_fused.sum(dim=1)
        sum_mask = mask.sum(dim=1).clamp(min=1e-9)
        fused_gene_cls = sum_fused / sum_mask  # [K, vision_dim]

        # Predict matching score and convert to probability
        igm_logits = self.igm_head(fused_gene_cls).squeeze(-1)  # [K]
        igm_probs = torch.sigmoid(igm_logits)

        return igm_probs

    @torch.no_grad()
    def get_gene_sequences(
        self,
        gene_sentences: List[str],
        max_length: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get gene sequence embeddings for caching (used in IGM inference).

        Returns:
            gene_cls: [B, gene_dim] mean-pooled
            gene_seq: [B, seq_len, gene_dim] sequence embeddings
            attention_mask: [B, seq_len]
        """
        return self.encode_gene(gene_sentences, max_length)

