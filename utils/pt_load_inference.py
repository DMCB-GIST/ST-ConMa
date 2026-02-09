"""
Inference utilities for loading pretrained models and extracting embeddings

This module provides functions to:
1. Load pretrained vision and gene models from checkpoints
2. Extract embeddings from images or gene sequences
3. Return embeddings as numpy arrays for downstream tasks
"""

import os
import torch
import numpy as np
import pandas as pd
import json
from typing import Union, List, Optional, Tuple, Dict
from pathlib import Path
from PIL import Image
from tqdm import tqdm

try:
    from .model import get_vision_model, get_c2s_model
    from .multimodal import ST_AlignmentModel
except ImportError:
    # Fall back to absolute imports when used as a script
    from model import get_vision_model, get_c2s_model
    from multimodal import ST_AlignmentModel


class VisionInference:
    """
    Vision model inference class

    Handles loading pretrained vision models and extracting image embeddings.
    Supports: pathoduet (768)
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model_name: str = 'pathoduet',
        device: str = 'cuda'
    ):
        
        self.device = torch.device(device)
        self.model_name = model_name

        self.transform, self.model = get_vision_model(
            model_name=model_name,
            device=self.device
        )

        # Load checkpoint if provided
        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)

        self.model.eval()
        print(f"Vision model ({model_name}) loaded for inference")

    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint weights"""
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

        # Extract vision_encoder weights from multimodal checkpoint
        vision_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('vision_encoder.'):
                # Remove 'vision_encoder.' prefix
                new_key = key.replace('vision_encoder.', '')
                vision_state_dict[new_key] = value

        if vision_state_dict:
            print(f"Loaded {len(vision_state_dict)} vision_encoder weights from multimodal checkpoint")
            state_dict = vision_state_dict

        # Load weights
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)

        if missing_keys:
            print(f"Warning: {len(missing_keys)} missing keys")
        if unexpected_keys:
            print(f"Warning: {len(unexpected_keys)} unexpected keys")

        print(f"Loaded checkpoint from: {checkpoint_path}")

    @torch.no_grad()
    def encode_image(
        self,
        image: Union[str, Path, Image.Image],
        return_numpy: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Extract embedding from a single image

        Args:
            image: image path, PIL Image, or tensor
            return_numpy: if True, return numpy array; else torch tensor

        Returns:
            embedding: [vision_dim] array or tensor
        """
        # Prepare image
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')

        if isinstance(image, Image.Image):
            image = self.transform(image)

        # Add batch dimension if needed
        if image.dim() == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device)

        # Extract embedding (wrapper returns patch_features, representation)
        patch_features, representation = self.model(image)  # [1, vision_dim]
        representation = representation.squeeze(0)  # [vision_dim]

        if return_numpy:
            return representation.cpu().numpy()
        return representation

    @torch.no_grad()
    def encode_images_batch(
        self,
        images: List[str],
        batch_size: int = 32,
        return_numpy: bool = True,
        save_csv: bool = False,
        output_dir: Optional[str] = None
    ) -> Union[np.ndarray, torch.Tensor, Tuple[np.ndarray, Dict[str, str]]]:
        """
        Extract embeddings from multiple images

        Args:
            images: list of image paths (format: {sample_id}_{spot_id}.png)
            batch_size: batch size for processing
            return_numpy: if True, return numpy array; else torch tensor
            save_csv: if True, save embeddings to CSV files grouped by sample_id
            output_dir: directory to save CSV files (required if save_csv=True)

        Returns:
            embeddings: [num_images, vision_dim] array or tensor
            If save_csv=True, also returns dict mapping sample_id to CSV path
        """
        # Parse image filenames to extract sample_id and spot_id
        image_info = []

        for img_path in images:
            filename = Path(img_path).stem
            parts = filename.split('_', 1)

            if len(parts) == 2:
                sample_id, spot_id = parts
            else:
                sample_id = "unknown"
                spot_id = filename

            image_info.append({
                'path': img_path,
                'sample_id': sample_id,
                'spot_id': spot_id
            })

        # Process list of images in batches
        all_embeddings = []

        for i in tqdm(range(0, len(images), batch_size), desc="Encoding images"):
            batch_images = images[i:i + batch_size]

            batch_tensors = []
            for img in batch_images:
                img = Image.open(img).convert('RGB')
                batch_tensors.append(self.transform(img))

            batch_tensor = torch.stack(batch_tensors).to(self.device)

            # Extract embeddings (wrapper returns patch_features, representation)
            _, batch_embeddings = self.model(batch_tensor)  # [B, vision_dim]
            all_embeddings.append(batch_embeddings)

        embeddings = torch.cat(all_embeddings, dim=0)

        # Save to CSV if requested
        if save_csv:
            if output_dir is None:
                raise ValueError("output_dir must be provided when save_csv=True")

            # Convert to numpy for saving
            embeddings_np = embeddings.cpu().numpy()

            # Group by sample_id
            from collections import defaultdict
            sample_groups = defaultdict(list)

            for idx, info in enumerate(image_info):
                sample_groups[info['sample_id']].append({
                    'spot_id': info['spot_id'],
                    'embedding': embeddings_np[idx]
                })

            # Save each sample_id to separate CSV
            os.makedirs(output_dir, exist_ok=True)
            saved_files = {}

            for sample_id, spots in sample_groups.items():
                # Create DataFrame for this sample
                df_data = {
                    'id': [spot['spot_id'] for spot in spots],
                    'embedding': [json.dumps(spot['embedding'].tolist()) for spot in spots]
                }
                df = pd.DataFrame(df_data)

                # Save to CSV
                csv_filename = f"{sample_id}_image_embeddings.csv"
                csv_path = os.path.join(output_dir, csv_filename)
                df.to_csv(csv_path, index=False)
                saved_files[sample_id] = csv_path
                print(f"Saved {len(spots)} embeddings for {sample_id} to {csv_path}")

            print(f"\nTotal: Saved {len(saved_files)} CSV files to {output_dir}")

            if return_numpy:
                return embeddings_np, saved_files
            return embeddings, saved_files

        if return_numpy:
            return embeddings.cpu().numpy()
        return embeddings


class GeneInference:
    """
    Gene model inference class

    Handles loading pretrained gene models (C2S pythia_410m) and extracting embeddings.
    Converts gene lists to sentences for C2S model input.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model_name: str = 'pythia_410m',
        device: str = 'cuda',
        max_length: int = 1024,
        layer_index: Optional[int] = None,
        num_layers: Optional[int] = None,
        random_init: bool = False
    ):
        """
        Initialize gene inference model

        Args:
            checkpoint_path: path to checkpoint file (None for pretrained only)
            model_name: C2S model name ('pythia_410m')
            device: device to run inference on
            max_length: maximum sequence length
            layer_index: which layer to extract hidden states from (None = last layer)
            num_layers: number of transformer layers to keep (None = all layers)
            random_init: if True, randomly initialize model weights instead of pretrained
        """
        self.device = torch.device(device)
        self.max_length = max_length
        self.layer_index = layer_index
        self.model_name = model_name
        self.random_init = random_init

        self.tokenizer, self.model = get_c2s_model(
            model_name=model_name,
            device=self.device,
            num_layers=num_layers,
            random_init=random_init
        )

        # Set pad_token if not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load checkpoint if provided
        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)

        self.model.eval()
        layer_info = f" - using layer {self.layer_index}" if self.layer_index is not None else " - using last layer"
        init_info = " [RANDOM INIT]" if random_init else ""
        print(f"Gene model (C2S: {model_name}) loaded for inference{layer_info}{init_info}")

    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint weights"""
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

        # Extract gene_encoder weights from multimodal checkpoint
        gene_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('gene_encoder.'):
                # Remove 'gene_encoder.' prefix
                new_key = key.replace('gene_encoder.', '')
                gene_state_dict[new_key] = value

        # If gene_encoder keys found, use them; otherwise use full state_dict
        if gene_state_dict:
            print(f"Loaded {len(gene_state_dict)} gene_encoder weights from multimodal checkpoint")
            state_dict = gene_state_dict

        # Load weights
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)

        if missing_keys:
            print(f"Warning: {len(missing_keys)} missing keys")
        if unexpected_keys:
            print(f"Warning: {len(unexpected_keys)} unexpected keys")

        print(f"Loaded checkpoint from: {checkpoint_path}")

    @torch.no_grad()
    def encode_gene_sentence(
        self,
        gene_sentence: str,
        return_numpy: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Extract embedding from a single gene sentence (space-separated gene names).

        Args:
            gene_sentence: space-separated gene names (e.g., 'VIM CD74 SPARC')
            return_numpy: if True, return numpy array; else torch tensor

        Returns:
            embedding: [gene_dim] array or tensor
        """
        # Tokenize
        tokens = self.tokenizer(
            gene_sentence,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_length
        )

        # Move to device
        input_ids = tokens['input_ids'].to(self.device)
        attention_mask = tokens['attention_mask'].to(self.device)

        # Extract embedding with mean pooling
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # Select layer based on layer_index
        if self.layer_index is None:
            hidden_states = outputs.hidden_states[-1]  # [1, L, H] - last layer
        else:
            hidden_states = outputs.hidden_states[self.layer_index]  # [1, L, H] - specific layer

        # Mean pooling (exclude padding)
        mask = attention_mask.unsqueeze(-1).float()
        masked_embeddings = hidden_states * mask
        sum_embeddings = masked_embeddings.sum(dim=1)  # [1, H]
        sum_mask = mask.sum(dim=1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)

        embedding = sum_embeddings / sum_mask  # [1, H]
        embedding = embedding.squeeze(0)  # [H]

        if return_numpy:
            return embedding.cpu().numpy()
        return embedding

    @torch.no_grad()
    def encode_gene_sentences_batch(
        self,
        gene_sentences: List[str],
        gene_ids: Optional[List[str]] = None,
        batch_size: int = 32,
        return_numpy: bool = True,
        save_csv: bool = False,
        output_dir: Optional[str] = None
    ) -> Union[np.ndarray, torch.Tensor, Tuple[np.ndarray, Dict[str, str]]]:
        """
        Extract embeddings from gene sentences (space-separated gene names).

        This method directly accepts gene sentences without unnecessary conversion,
        matching the training data format (e.g., 'VIM CD74 SPARC FTL ...').

        Args:
            gene_sentences: list of space-separated gene sentences
            gene_ids: list of IDs for each sentence (format: {sample_id}_{spot_id})
            batch_size: batch size for processing
            return_numpy: if True, return numpy array; else torch tensor
            save_csv: if True, save embeddings to CSV files grouped by sample_id
            output_dir: directory to save CSV files (required if save_csv=True)

        Returns:
            embeddings: [num_sentences, gene_dim] array or tensor
            If save_csv=True, also returns dict mapping sample_id to CSV path
        """
        # Parse gene IDs to extract sample_id and spot_id
        if gene_ids is not None:
            gene_info = []

            for gene_id in gene_ids:
                parts = gene_id.split('_', 1)

                if len(parts) == 2:
                    sample_id, spot_id = parts
                else:
                    sample_id = "unknown"
                    spot_id = gene_id

                gene_info.append({
                    'sample_id': sample_id,
                    'spot_id': spot_id
                })
        else:
            # If no IDs provided, use index as spot_id
            gene_info = [
                {'sample_id': 'unknown', 'spot_id': f'spot_{i}'}
                for i in range(len(gene_sentences))
            ]

        all_embeddings = []

        for i in tqdm(range(0, len(gene_sentences), batch_size), desc="Encoding gene sentences"):
            batch_sentences = gene_sentences[i:i + batch_size]

            # Tokenize batch directly (no conversion needed)
            tokens = self.tokenizer(
                batch_sentences,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=self.max_length
            )

            input_ids = tokens['input_ids'].to(self.device)
            attention_mask = tokens['attention_mask'].to(self.device)

            # Extract embeddings with mean pooling
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )

            # Select layer based on layer_index
            if self.layer_index is None:
                hidden_states = outputs.hidden_states[-1]  # [B, L, H] - last layer
            else:
                hidden_states = outputs.hidden_states[self.layer_index]  # [B, L, H] - specific layer

            # Mean pooling (exclude padding)
            mask = attention_mask.unsqueeze(-1).float()
            masked_embeddings = hidden_states * mask
            sum_embeddings = masked_embeddings.sum(dim=1)  # [B, H]
            sum_mask = mask.sum(dim=1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)

            batch_embeddings = sum_embeddings / sum_mask  # [B, H]
            all_embeddings.append(batch_embeddings)

        embeddings = torch.cat(all_embeddings, dim=0)

        # Save to CSV if requested
        if save_csv:
            if output_dir is None:
                raise ValueError("output_dir must be provided when save_csv=True")

            # Convert to numpy for saving
            embeddings_np = embeddings.cpu().numpy()

            # Group by sample_id
            from collections import defaultdict
            sample_groups = defaultdict(list)

            for idx, info in enumerate(gene_info):
                sample_groups[info['sample_id']].append({
                    'spot_id': info['spot_id'],
                    'embedding': embeddings_np[idx]
                })

            # Save each sample_id to separate CSV
            os.makedirs(output_dir, exist_ok=True)
            saved_files = {}

            for sample_id, spots in sample_groups.items():
                # Create DataFrame for this sample
                df_data = {
                    'id': [spot['spot_id'] for spot in spots],
                    'embedding': [json.dumps(spot['embedding'].tolist()) for spot in spots]
                }
                df = pd.DataFrame(df_data)

                # Save to CSV
                csv_filename = f"{sample_id}_gene_embeddings.csv"
                csv_path = os.path.join(output_dir, csv_filename)
                df.to_csv(csv_path, index=False)
                saved_files[sample_id] = csv_path
                print(f"Saved {len(spots)} gene embeddings for {sample_id} to {csv_path}")

            print(f"\nTotal: Saved {len(saved_files)} gene embedding CSV files to {output_dir}")

            if return_numpy:
                return embeddings_np, saved_files
            return embeddings, saved_files

        if return_numpy:
            return embeddings.cpu().numpy()
        return embeddings


class MultimodalInference:
    """
    Multimodal model inference class

    Handles loading pretrained ST-ConMa-text models (vision + gene encoders + projection heads)
    and extracting projected embeddings for IGC (Image-Gene Contrastive).
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        vision_model_name: str = 'pathoduet',
        gene_model_name: str = 'pythia_410m',
        device: str = 'cuda',
        vision_dim: int = 768,
        gene_dim: int = 1024,  # pythia_410m: 1024
        proj_dim: int = 768,
        max_seq_len: int = 1024,
        proj_hidden_dim: int = 3072,
        proj_layers: int = 2,
        num_cross_layers: int = 3,
        num_heads: int = 12,
        loss_type: str = "clip",
        temperature: float = 0.07,
        learnable_temperature: bool = False,
        # Multi-task loss weights
        igc_weight: float = 1.0,
        iic_weight: float = 1.0,
        igm_weight: float = 1.0,
        # Individual loss configurations
        igc_loss_type: Optional[str] = None,
        iic_loss_type: Optional[str] = None,
        igc_temperature: Optional[float] = None,
        iic_temperature: Optional[float] = None,
        # Layer selection for C2S model
        num_layers: Optional[int] = 12,
        layer_index: Optional[int] = None
    ):
        """
        Initialize multimodal inference model with C2S gene encoder

        Args:
            checkpoint_path: path to multimodal checkpoint file (None = random initialization)
            vision_model_name: vision model name ('pathoduet')
            gene_model_name: C2S model name ('pythia_410m')
            device: device to run inference on
            vision_dim: vision encoder output dimension
            gene_dim: gene encoder output dimension (pythia_410m: 1024)
            proj_dim: projection dimension
            max_seq_len: maximum sequence length
            proj_hidden_dim: projection head hidden dimension
            proj_layers: number of projection layers
            num_cross_layers: number of cross-attention layers for IGM
            num_heads: number of attention heads
            loss_type: default loss type ('clip' or 'siglip') - used as fallback for igc_loss_type and iic_loss_type
            temperature: default temperature for contrastive loss - used as fallback for igc_temperature and iic_temperature
            learnable_temperature: whether temperature is learnable
            igc_weight: weight for Image-Gene Contrastive loss
            iic_weight: weight for Image-Image Contrastive loss
            igm_weight: weight for Image-Gene Matching loss
            igc_loss_type: loss type specifically for IGC (None = use loss_type)
            iic_loss_type: loss type specifically for IIC (None = use loss_type)
            igc_temperature: temperature specifically for IGC (None = use temperature)
            iic_temperature: temperature specifically for IIC (None = use temperature)
            num_layers: number of transformer layers to keep in gene encoder (None = use all layers)
            layer_index: which layer to extract from C2S model (None = last layer)
        """
        self.device = torch.device(device)
        self.vision_model_name = vision_model_name
        self.max_seq_len = max_seq_len

        # Load vision encoder
        print(f"Loading vision encoder: {vision_model_name}")
        self.transform, vision_encoder = get_vision_model(
            model_name=vision_model_name,
            device=self.device
        )

        # Load gene encoder (C2S model)
        print(f"Loading gene encoder: C2S model ({gene_model_name})")
        self.tokenizer, gene_encoder = get_c2s_model(
            model_name=gene_model_name,
            device=self.device,
            num_layers=num_layers
        )

        if num_layers is not None:
            print(f"  Using only first {num_layers} layers of gene encoder")

        # Set pad_token if not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.gene_model_name = gene_model_name

        # Create multimodal model
        print("Creating multimodal model (ST-ConMa-text architecture)")
        self.model = ST_AlignmentModel(
            vision_encoder=vision_encoder,
            gene_encoder=gene_encoder,
            tokenizer=self.tokenizer,
            vision_dim=vision_dim,
            gene_dim=gene_dim,
            proj_dim=proj_dim,
            proj_hidden_dim=proj_hidden_dim,
            proj_layers=proj_layers,
            num_cross_layers=num_cross_layers,
            num_heads=num_heads,
            freeze_encoders=False,
            loss_type=loss_type,
            temperature=temperature,
            learnable_temperature=learnable_temperature,
            igc_weight=igc_weight,
            iic_weight=iic_weight,
            igm_weight=igm_weight,
            igc_loss_type=igc_loss_type,
            iic_loss_type=iic_loss_type,
            igc_temperature=igc_temperature,
            iic_temperature=iic_temperature,
            layer_index=layer_index,
            device=self.device
        ).to(self.device)

        # Load checkpoint if provided
        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)
            print(f"Multimodal model loaded from checkpoint")
        else:
            print(f"Multimodal model initialized with random weights (no checkpoint loaded)")

        self.model.eval()

    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint weights"""
        print(f"Loading checkpoint: {checkpoint_path}")
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

        # Load weights
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)

        # Count successfully loaded keys
        num_loaded = len(state_dict) - len(unexpected_keys)
        print(f"  Loaded {num_loaded}/{len(state_dict)} keys from checkpoint")

        if missing_keys:
            print(f"  Warning: Missing {len(missing_keys)} keys")

        if unexpected_keys:
            print(f"  Warning: {len(unexpected_keys)} unexpected keys")

        print(f"Checkpoint loaded from: {checkpoint_path}")

    @torch.no_grad()
    def encode_vision(
        self,
        images: torch.Tensor,
        return_numpy: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Extract vision embeddings through projection head (for IGC)

        Args:
            images: batch of images [B, C, H, W]
            return_numpy: if True, return numpy array

        Returns:
            projected embeddings [B, proj_dim]
        """
        images = images.to(self.device)

        # Get vision representation
        vision_representation, _ = self.model.encode_vision(images)

        # Project to shared space
        vision_projs = self.model.vision_projection(vision_representation)

        if return_numpy:
            return vision_projs.cpu().numpy()
        return vision_projs

    @torch.no_grad()
    def encode_gene(
        self,
        gene_sentences: List[str],
        return_numpy: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Extract gene embeddings through projection head (for IGC)

        Args:
            gene_sentences: list of space-separated gene sentences
            return_numpy: if True, return numpy array

        Returns:
            projected embeddings [B, proj_dim]
        """
        # Get gene representation
        gene_cls, _, _ = self.model.encode_gene(gene_sentences, max_length=self.max_seq_len)

        # Project to shared space
        gene_projs = self.model.gene_projection(gene_cls)

        if return_numpy:
            return gene_projs.cpu().numpy()
        return gene_projs

    @torch.no_grad()
    def encode_images_batch(
        self,
        images: List[str],
        batch_size: int = 32,
        return_numpy: bool = True,
        save_csv: bool = False,
        output_dir: Optional[str] = None
    ) -> Union[np.ndarray, torch.Tensor, Tuple[np.ndarray, Dict[str, str]]]:
        """
        Extract projected vision embeddings from multiple images

        Args:
            images: list of image paths (format: {sample_id}_{spot_id}.png)
            batch_size: batch size for processing
            return_numpy: if True, return numpy array; else torch tensor
            save_csv: if True, save embeddings to CSV files grouped by sample_id
            output_dir: directory to save CSV files (required if save_csv=True)

        Returns:
            embeddings: [num_images, proj_dim] array or tensor
            If save_csv=True, also returns dict mapping sample_id to CSV path
        """
        # Parse image filenames to extract sample_id and spot_id
        image_info = []

        for img_path in images:
            filename = Path(img_path).stem
            parts = filename.split('_', 1)

            if len(parts) == 2:
                sample_id, spot_id = parts
            else:
                sample_id = "unknown"
                spot_id = filename

            image_info.append({
                'path': img_path,
                'sample_id': sample_id,
                'spot_id': spot_id
            })

        # Process list of images in batches
        all_embeddings = []

        for i in tqdm(range(0, len(images), batch_size), desc="Encoding images (projected)"):
            batch_images = images[i:i + batch_size]

            # Load and transform images
            batch_tensors = []
            for img in batch_images:
                img = Image.open(img).convert('RGB')
                batch_tensors.append(self.transform(img))

            batch_tensor = torch.stack(batch_tensors).to(self.device)

            # Extract embeddings through projection head
            batch_embeddings = self.encode_vision(batch_tensor, return_numpy=False)
            all_embeddings.append(batch_embeddings)

        embeddings = torch.cat(all_embeddings, dim=0)

        # Save to CSV if requested
        if save_csv:
            if output_dir is None:
                raise ValueError("output_dir must be provided when save_csv=True")

            # Convert to numpy for saving
            embeddings_np = embeddings.cpu().numpy()

            # Group by sample_id
            from collections import defaultdict
            sample_groups = defaultdict(list)

            for idx, info in enumerate(image_info):
                sample_groups[info['sample_id']].append({
                    'spot_id': info['spot_id'],
                    'embedding': embeddings_np[idx]
                })

            # Save each sample_id to separate CSV
            os.makedirs(output_dir, exist_ok=True)
            saved_files = {}

            for sample_id, spots in sample_groups.items():
                # Create DataFrame for this sample
                df_data = {
                    'id': [spot['spot_id'] for spot in spots],
                    'embedding': [json.dumps(spot['embedding'].tolist()) for spot in spots]
                }
                df = pd.DataFrame(df_data)

                # Save to CSV
                csv_filename = f"{sample_id}_image_projected_embeddings.csv"
                csv_path = os.path.join(output_dir, csv_filename)
                df.to_csv(csv_path, index=False)
                saved_files[sample_id] = csv_path
                print(f"Saved {len(spots)} projected embeddings for {sample_id} to {csv_path}")

            print(f"\nTotal: Saved {len(saved_files)} CSV files to {output_dir}")

            if return_numpy:
                return embeddings_np, saved_files
            return embeddings, saved_files

        if return_numpy:
            return embeddings.cpu().numpy()
        return embeddings

    @torch.no_grad()
    def encode_gene_sentences_batch(
        self,
        gene_sentences: List[str],
        gene_ids: Optional[List[str]] = None,
        batch_size: int = 32,
        return_numpy: bool = True,
        save_csv: bool = False,
        output_dir: Optional[str] = None
    ) -> Union[np.ndarray, torch.Tensor, Tuple[np.ndarray, Dict[str, str]]]:
        """
        Extract projected gene embeddings from multiple gene sentences

        Args:
            gene_sentences: list of space-separated gene sentences
            gene_ids: list of IDs for each sentence (format: {sample_id}_{spot_id})
            batch_size: batch size for processing
            return_numpy: if True, return numpy array; else torch tensor
            save_csv: if True, save embeddings to CSV files grouped by sample_id
            output_dir: directory to save CSV files (required if save_csv=True)

        Returns:
            embeddings: [num_sentences, proj_dim] array or tensor
            If save_csv=True, also returns dict mapping sample_id to CSV path
        """
        # Parse gene IDs to extract sample_id and spot_id
        if gene_ids is not None:
            gene_info = []

            for gene_id in gene_ids:
                parts = gene_id.split('_', 1)

                if len(parts) == 2:
                    sample_id, spot_id = parts
                else:
                    sample_id = "unknown"
                    spot_id = gene_id

                gene_info.append({
                    'sample_id': sample_id,
                    'spot_id': spot_id
                })
        else:
            # If no IDs provided, use index as spot_id
            gene_info = [
                {'sample_id': 'unknown', 'spot_id': f'spot_{i}'}
                for i in range(len(gene_sentences))
            ]

        all_embeddings = []

        for i in tqdm(range(0, len(gene_sentences), batch_size), desc="Encoding gene sentences (projected)"):
            batch_gene_sentences = gene_sentences[i:i + batch_size]

            # Extract embeddings through projection head
            batch_embeddings = self.encode_gene(batch_gene_sentences, return_numpy=False)
            all_embeddings.append(batch_embeddings)

        embeddings = torch.cat(all_embeddings, dim=0)

        # Save to CSV if requested
        if save_csv:
            if output_dir is None:
                raise ValueError("output_dir must be provided when save_csv=True")

            # Convert to numpy for saving
            embeddings_np = embeddings.cpu().numpy()

            # Group by sample_id
            from collections import defaultdict
            sample_groups = defaultdict(list)

            for idx, info in enumerate(gene_info):
                sample_groups[info['sample_id']].append({
                    'spot_id': info['spot_id'],
                    'embedding': embeddings_np[idx]
                })

            # Save each sample_id to separate CSV
            os.makedirs(output_dir, exist_ok=True)
            saved_files = {}

            for sample_id, spots in sample_groups.items():
                # Create DataFrame for this sample
                df_data = {
                    'id': [spot['spot_id'] for spot in spots],
                    'embedding': [json.dumps(spot['embedding'].tolist()) for spot in spots]
                }
                df = pd.DataFrame(df_data)

                # Save to CSV
                csv_filename = f"{sample_id}_gene_projected_embeddings.csv"
                csv_path = os.path.join(output_dir, csv_filename)
                df.to_csv(csv_path, index=False)
                saved_files[sample_id] = csv_path
                print(f"Saved {len(spots)} projected gene embeddings for {sample_id} to {csv_path}")

            print(f"\nTotal: Saved {len(saved_files)} gene embedding CSV files to {output_dir}")

            if return_numpy:
                return embeddings_np, saved_files
            return embeddings, saved_files

        if return_numpy:
            return embeddings.cpu().numpy()
        return embeddings

    @torch.no_grad()
    def encode_fused(
        self,
        images: torch.Tensor,
        gene_sentences: List[str],
        return_numpy: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Extract fused embeddings after cross-attention (before IGM head)

        This is the embedding that goes into the IGM head for matching prediction.
        Gene features are fused with image features via cross-attention.

        Args:
            images: batch of images [B, C, H, W]
            gene_sentences: list of space-separated gene sentences
            return_numpy: if True, return numpy array

        Returns:
            fused embeddings [B, vision_dim] - after cross-attention, before IGM head
        """
        images = images.to(self.device)

        # Get vision patches (not the representation)
        # encode_vision returns (vision_representation, vision_patches)
        _, vision_patches = self.model.encode_vision(images)  # [B, num_patches, vision_dim]

        # Get gene sequence features
        _, gene_seq, gene_attn_mask = self.model.encode_gene(
            gene_sentences, max_length=self.max_seq_len
        )  # gene_seq: [B, seq_len, gene_dim]

        # Align gene dimension to vision dimension
        gene_seq_aligned = self.model.gene_dim_proj(gene_seq)  # [B, seq_len, vision_dim]

        # Cross-attention fusion: Gene ← Image
        fused_gene = self.model.cross_attention(
            gene_features=gene_seq_aligned,
            image_features=vision_patches,
            gene_attn_mask=gene_attn_mask,
            image_attn_mask=None
        ) 

        # Average pooling across sequence (excluding padding)
        mask = gene_attn_mask.unsqueeze(-1).float()  # [B, seq_len, 1]
        masked_fused = fused_gene * mask  # [B, seq_len, vision_dim]
        sum_fused = masked_fused.sum(dim=1)  # [B, vision_dim]
        sum_mask = mask.sum(dim=1)  # [B, 1]
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        fused_gene_cls = sum_fused / sum_mask  # [B, vision_dim]

        if return_numpy:
            return fused_gene_cls.cpu().numpy()
        return fused_gene_cls

    @torch.no_grad()
    def encode_fused_batch(
        self,
        images: List[str],
        gene_sentences: List[str],
        gene_ids: Optional[List[str]] = None,
        batch_size: int = 32,
        return_numpy: bool = True,
        save_csv: bool = False,
        output_dir: Optional[str] = None
    ) -> Union[np.ndarray, torch.Tensor, Tuple[np.ndarray, Dict[str, str]]]:
        """
        Extract fused embeddings (after cross-attention, before IGM head) from multiple image-gene pairs

        Returns:
            embeddings: [num_pairs, vision_dim] array or tensor
            If save_csv=True, also returns dict mapping sample_id to CSV path
        """
        assert len(images) == len(gene_sentences), "images and gene_sentences must have same length"

        # Parse IDs to extract sample_id and spot_id
        if gene_ids is not None:
            pair_info = []
            for gene_id in gene_ids:
                parts = gene_id.split('_', 1)
                if len(parts) == 2:
                    sample_id, spot_id = parts
                else:
                    sample_id = "unknown"
                    spot_id = gene_id
                pair_info.append({'sample_id': sample_id, 'spot_id': spot_id})
        else:
            pair_info = [
                {'sample_id': 'unknown', 'spot_id': f'spot_{i}'}
                for i in range(len(images))
            ]

        all_embeddings = []

        for i in tqdm(range(0, len(images), batch_size), desc="Encoding fused embeddings"):
            batch_images = images[i:i + batch_size]
            batch_gene_sentences = gene_sentences[i:i + batch_size]

            # Load and transform images
            batch_tensors = []
            for img in batch_images:
                img = Image.open(img).convert('RGB')
                batch_tensors.append(self.transform(img))

            batch_tensor = torch.stack(batch_tensors).to(self.device)

            # Extract fused embeddings
            batch_embeddings = self.encode_fused(
                batch_tensor, batch_gene_sentences, return_numpy=False
            )
            all_embeddings.append(batch_embeddings)

        embeddings = torch.cat(all_embeddings, dim=0)

        # Save to CSV if requested
        if save_csv:
            if output_dir is None:
                raise ValueError("output_dir must be provided when save_csv=True")

            embeddings_np = embeddings.cpu().numpy()

            from collections import defaultdict
            sample_groups = defaultdict(list)

            for idx, info in enumerate(pair_info):
                sample_groups[info['sample_id']].append({
                    'spot_id': info['spot_id'],
                    'embedding': embeddings_np[idx]
                })

            os.makedirs(output_dir, exist_ok=True)
            saved_files = {}

            for sample_id, spots in sample_groups.items():
                df_data = {
                    'id': [spot['spot_id'] for spot in spots],
                    'embedding': [json.dumps(spot['embedding'].tolist()) for spot in spots]
                }
                df = pd.DataFrame(df_data)

                csv_filename = f"{sample_id}_fused_embeddings.csv"
                csv_path = os.path.join(output_dir, csv_filename)
                df.to_csv(csv_path, index=False)
                saved_files[sample_id] = csv_path
                print(f"Saved {len(spots)} fused embeddings for {sample_id} to {csv_path}")

            print(f"\nTotal: Saved {len(saved_files)} fused embedding CSV files to {output_dir}")

            if return_numpy:
                return embeddings_np, saved_files
            return embeddings, saved_files

        if return_numpy:
            return embeddings.cpu().numpy()
        return embeddings


# ============================================================================
# Convenience functions
# ============================================================================

def load_vision_encoder(
    checkpoint_path: Optional[str] = None,
    model_name: str = 'pathoduet',
    device: str = 'cuda'
) -> VisionInference:
    """
    Returns:
        VisionInference instance
    """
    return VisionInference(
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        device=device
    )


def load_gene_encoder(
    checkpoint_path: Optional[str] = None,
    model_name: str = 'pythia_410m',
    device: str = 'cuda',
    max_length: int = 512,
    layer_index: Optional[int] = None
) -> GeneInference:
    """
    Returns:
        GeneInference instance
    """
    return GeneInference(
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        device=device,
        max_length=max_length,
        layer_index=layer_index
    )


def load_multimodal_model(
    checkpoint_path: str,
    vision_model_name: str = 'pathoduet',
    gene_model_name: str = 'pythia_410m',
    device: str = 'cuda',
    **kwargs
) -> MultimodalInference:
    """
    Returns:
        MultimodalInference instance
    """
    return MultimodalInference(
        checkpoint_path=checkpoint_path,
        vision_model_name=vision_model_name,
        gene_model_name=gene_model_name,
        device=device,
        **kwargs
    )
