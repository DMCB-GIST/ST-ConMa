import os
import sys
from typing import Tuple, Optional, Union
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedModel
sys.path.append("./PathoDuet")
from vits import VisionTransformerMoCo
import torchvision.transforms as transforms



class PathoDuetWrapper(nn.Module):
    """
    Wrapper for PathoDuet model to extract patch features.

    PathoDuet returns a tuple (patch_features, pooled_embedding).
    We return patch_features for multi-modal fusion with cross-attention.
    """

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        """
        Args:
            x: input images [batch_size, 3, H, W]

        Returns:
            patch_features: [batch_size, num_patches, embed_dim]
        """
        # Get patch features from base model (first element of tuple)
        # x_out: [batch, num_patches, 768], x: [batch, 768]
        patch_features, representation = self.base_model(x)
        patch_features = patch_features[:, 2:, :]
        return patch_features, representation


def get_vision_model(
    model_name: str,
    device: Union[str, torch.device],
) -> Tuple[object, torch.nn.Module]:
    """
    Load vision model.

    Args:
        model_name: name of the vision model ('pathoduet')
        device: device to load model on

    Returns:
        transform: data transform for the model
        model: loaded model
            - pathoduet: output dim = 768

    Raises:
        ValueError: if model_name is not supported
    """
    if model_name == "pathoduet":
        # Load PathoDuet base model
        base_model = VisionTransformerMoCo(pretext_token=True, global_pool="avg")

        # Load checkpoint
        checkpoint = torch.load("./PathoDuet/checkpoint/checkpoint_HE.pth", map_location=device)
        base_model.load_state_dict(checkpoint, strict=False)

        # Remove head (we only need embeddings)
        base_model.head = torch.nn.Identity()

        # Wrap with PathoDuetWrapper to properly extract embeddings
        model = PathoDuetWrapper(base_model).to(device)

        # Define transform for PathoDuet
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    else:
        raise ValueError(
            f"Unsupported vision model: {model_name}. "
            f"Supported models: ['pathoduet']"
        )

    print(f"Vision model loaded: {model_name}")

    return transform, model


def get_c2s_model(
    model_name: str,
    device: Union[str, torch.device],
    num_layers: Optional[int] = None,
    random_init: bool = False
    ) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
    """
    Load C2S (Cell-to-Sentence) model.

    Args:
        model_name: name of the C2S model ('pythia_410m')
        device: device to load model on
        num_layers: if specified, only keep the first num_layers transformer layers (reduces model size)
        random_init: if True, randomly initialize model weights instead of using pretrained weights

    Returns:
        tokenizer: tokenizer for the model
        model: loaded model

    Raises:
        ValueError: if model_name is not supported
    """
    from transformers import AutoConfig

    if model_name == "pythia_410m":
        model_id = "vandijklab/C2S-Pythia-410m-diverse-single-and-multi-cell-tasks"

        # Load tokenizer (requires sentencepiece)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        if random_init:
            # Random initialization: load config only and create model from scratch
            config = AutoConfig.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_config(config).to(device)
            print(f"Pythia-410m loaded with RANDOM initialization")
        else:
            model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

    else:
        raise ValueError(
            f"Unsupported C2S model: {model_name}. "
            f"Supported models: ['pythia_410m']"
        )

    # Truncate layers if specified (reduces model size)
    if num_layers is not None:
        original_num_layers = len(model.gpt_neox.layers)
        if num_layers < original_num_layers:
            # Keep only first num_layers transformer layers
            model.gpt_neox.layers = model.gpt_neox.layers[:num_layers]
            print(f"Truncated model from {original_num_layers} to {num_layers} layers")
        elif num_layers > original_num_layers:
            print(f"Warning: Requested {num_layers} layers but model only has {original_num_layers} layers")

    print(f"C2S model loaded: {model_name}")

    return tokenizer, model


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """
    Count model parameters.

    Args:
        model: PyTorch model
        trainable_only: if True, count only trainable parameters

    Returns:
        number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def print_model_info(model: torch.nn.Module, model_name: str = "Model") -> None:
    """
    Print model information including parameter counts.

    Args:
        model: PyTorch model
        model_name: name to display
    """
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    frozen_params = total_params - trainable_params

    print(f"\n{'='*50}")
    print(f"{model_name} Information:")
    print(f"{'='*50}")
    print(f"Total parameters:      {total_params:,}")
    print(f"Trainable parameters:  {trainable_params:,}")
    print(f"Frozen parameters:     {frozen_params:,}")
    print(f"Trainable ratio:       {trainable_params/total_params*100:.2f}%")
    print(f"{'='*50}\n")
