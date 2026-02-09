import torch
import torch.distributed as dist
from typing import List, Optional


class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all processes, supporting backward propagation.
    """

    @staticmethod
    def forward(ctx, input):
        """
        Forward pass: gather tensors from all GPUs

        Args:
            input: tensor to gather [batch_size, feature_dim]

        Returns:
            gathered tensor [world_size * batch_size, feature_dim]
        """
        ctx.save_for_backward(input) 

        if not dist.is_available() or not dist.is_initialized():
            return input

        # Get world size and rank
        world_size = dist.get_world_size()

        # Create placeholder for gathering
        gather_list = [torch.zeros_like(input) for _ in range(world_size)]

        # All gather
        dist.all_gather(gather_list, input)

        # Concatenate along batch dimension
        output = torch.cat(gather_list, dim=0)

        return output

    @staticmethod
    def backward(ctx, grad_output):

        input, = ctx.saved_tensors

        if not dist.is_available() or not dist.is_initialized():
            return grad_output

        # Get rank
        rank = dist.get_rank()

        # Get the batch size per GPU
        batch_size = input.shape[0]

        # Split the gradient back to each process
        # Each process gets its own portion of the gradient
        grad_input = grad_output[rank * batch_size : (rank + 1) * batch_size]

        return grad_input


def all_gather_with_grad(tensor):
    """
    Gather tensors from all processes with gradient support.

    Returns:
        gathered tensor [world_size * batch_size, feature_dim]
    """
    return GatherLayer.apply(tensor)


def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    WARNING: This function does NOT support gradient backpropagation.
    Use this only for inference or when gradients are not needed.

    Args:
        tensor: input tensor

    Returns:
        concatenated tensor from all processes
    """
    if not dist.is_available() or not dist.is_initialized():
        return tensor

    world_size = dist.get_world_size()

    tensors_gather = [torch.ones_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def gather_embeddings_for_contrastive(query_emb, key_emb, labels=None):
    """
    Gather embeddings from all GPUs for contrastive learning.

    Returns:
        all_query_emb: [world_size * batch_size, feature_dim]
        all_key_emb: [world_size * batch_size, feature_dim]
        all_labels: [world_size * batch_size] or None
        rank_offset: offset for computing local indices
    """
    if not dist.is_available() or not dist.is_initialized():
        if labels is not None:
            return query_emb, key_emb, labels, 0
        return query_emb, key_emb, None, 0

    # Gather embeddings with gradient support
    all_query_emb = all_gather_with_grad(query_emb)
    all_key_emb = all_gather_with_grad(key_emb)

    # Gather labels without gradient (labels don't need gradients)
    all_labels = None
    if labels is not None:
        all_labels = concat_all_gather(labels)

    # Calculate offset for this rank
    rank = dist.get_rank()
    batch_size = query_emb.shape[0]
    rank_offset = rank * batch_size

    return all_query_emb, all_key_emb, all_labels, rank_offset