import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np

class CLIPLoss(nn.Module):
    """
    CLIP-style Bidirectional Contrastive Loss

    Learns aligned representations between two modalities (e.g., image-text)
    by computing contrastive loss in both directions.
    """

    def __init__(self, temperature=0.07, learnable_temperature=True):
        """
        Args:
            temperature: temperature parameter for scaling logits (default: 0.07)
            learnable_temperature: if True, temperature is a learnable parameter
        """
        super(CLIPLoss, self).__init__()
        
        if learnable_temperature:
            # Store as logit_scale in log space
            # logit_scale = log(1/temperature)
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1.0 / temperature))
        else:
            self.register_buffer('logit_scale', torch.tensor(np.log(1.0 / temperature)))

    def forward(self, modality1_features, modality2_features):
        """
        Compute bidirectional contrastive loss.

        Args:
            modality1_features: features from modality 1 (e.g., image) [batch_size, feature_dim]
            modality2_features: features from modality 2 (e.g., text) [batch_size, feature_dim]

        """
        # Normalize features
        modality1_features = F.normalize(modality1_features, dim=-1)
        modality2_features = F.normalize(modality2_features, dim=-1)

        # Get logit scale (equivalent to 1/temperature)
        logit_scale = self.logit_scale.exp()

        logits_per_modality1 = torch.matmul(modality1_features, modality2_features.T) * logit_scale

        logits_per_modality2 = logits_per_modality1.T

        # Ground truth: diagonal elements are positive pairs
        batch_size = modality1_features.shape[0]
        labels = torch.arange(batch_size, device=modality1_features.device)

        # Compute cross-entropy loss in both directions
        loss_modality1_to_modality2 = F.cross_entropy(logits_per_modality1, labels)
        loss_modality2_to_modality1 = F.cross_entropy(logits_per_modality2, labels)

        # Average loss from both directions
        loss = (loss_modality1_to_modality2 + loss_modality2_to_modality1) / 2

        return loss


class SigLIPLoss(nn.Module):
    """
    SigLIP Loss (Sigmoid Loss for Language-Image Pre-training)
    """

    def __init__(self, temperature=10.0, bias=0.0, learnable_temperature=True, learnable_bias=False,
                 use_ring_chunked=False, local_loss=False, ring_use_barrier=True):
        """
        Args:
            temperature: initial temperature parameter (default: 10.0)
                Note: SigLIP multiplies by temperature (unlike CLIP which divides)
            bias: initial bias term for sigmoid
            learnable_temperature: if True, temperature is a learnable parameter
            learnable_bias: if True, bias is a learnable parameter
            use_ring_chunked: if True, use ring-based collective permute for distributed training
                This is the method described in SigLIP paper Section 3.3
            local_loss: if True, only compute loss for local batch (no gathering)
            ring_use_barrier: if True, use barriers for safer ring communication (slower but safer)
        """
        super(SigLIPLoss, self).__init__()

        if learnable_temperature:
            # Store in log space to ensure positivity
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(temperature))
        else:
            self.register_buffer('logit_scale', torch.tensor(np.log(temperature)))

        if learnable_bias:
            self.bias = nn.Parameter(torch.ones([]) * bias)
        else:
            self.register_buffer('bias', torch.tensor(bias))

        self.use_ring_chunked = use_ring_chunked
        self.local_loss = local_loss
        self.ring_use_barrier = ring_use_barrier

    def forward(self, modality1_features, modality2_features):
        """
        Args:
            modality1_features: modality1 features [batch_size, feature_dim]
            modality2_features: modality2 features [batch_size, feature_dim]

        Returns:
            SigLIP loss
        """
        # Normalize
        modality1_features = F.normalize(modality1_features, dim=-1)
        modality2_features = F.normalize(modality2_features, dim=-1)

        # Get temperature (exp to ensure positive)
        temperature = self.logit_scale.exp()

        # Choose implementation based on configuration
        if self.local_loss:
            # Only compute loss on local batch (no gathering)
            return self._local_forward(modality1_features, modality2_features, temperature)
        elif self.use_ring_chunked and dist.is_available() and dist.is_initialized():
            # Ring-based chunked implementation for distributed training
            return self._ring_chunked_forward(modality1_features, modality2_features, temperature)
        else:
            # Standard implementation
            return self._standard_forward(modality1_features, modality2_features, temperature)

    def _standard_forward(self, modality1_features, modality2_features, temperature):
        """Standard forward pass (computes full similarity matrix at once)"""
        batch_size = modality1_features.shape[0]

        # Compute similarity matrix and scale by temperature
        logits = torch.matmul(modality1_features, modality2_features.T) * temperature

        # Create labels: 1 for positive pairs (diagonal), -1 for negative pairs
        labels = torch.eye(batch_size, device=modality1_features.device) * 2 - 1

        # Sigmoid loss with optional bias
        loss = -F.logsigmoid(labels * (logits + self.bias))

        # Average over all pairs
        loss = loss.mean()

        return loss

    def _local_forward(self, modality1_features, modality2_features, temperature):
        """Compute loss only on local batch (for debugging/testing)"""
        return self._standard_forward(modality1_features, modality2_features, temperature)

    def _ring_chunked_forward(self, modality1_features, modality2_features, temperature):
        """
        Ring-based chunked implementation from SigLIP paper.

        This method uses collective permute (ring communication) to compute the loss
        efficiently in a distributed setting. Each device computes loss with respect
        to its local batch b, while cycling through embeddings from all other devices.

        Algorithm:
        1. Compute loss for positive pairs and local negatives
        2. For each of D-1 iterations:
           - Permute text_features to next device (ring pattern)
           - Compute loss for local images vs permuted texts (all negatives)
        3. Sum losses across all devices
        """
        if not dist.is_available() or not dist.is_initialized():
            # Fallback to standard if not in distributed mode
            return self._standard_forward(modality1_features, modality2_features, temperature)

        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_batch_size = modality1_features.shape[0]

        scaled_bias = self.bias

        total_loss = 0.0
        num_loss_chunks = 0

        # Step 1: Compute loss for local positive pairs and local negative pairs
        # This is the diagonal block: contains positives on diagonal
        logits_local = torch.matmul(modality1_features, modality2_features.T) * temperature

        # Labels: 1 for diagonal (positive), -1 for off-diagonal (negative)
        labels_local = torch.eye(local_batch_size, device=modality1_features.device) * 2 - 1

        # Compute loss for local chunk
        loss_local = -F.logsigmoid(labels_local * (logits_local + scaled_bias))
        total_loss += loss_local.sum()
        num_loss_chunks += local_batch_size * local_batch_size

        modality2_features_permuted = modality2_features.clone()

        for step in range(1, world_size):
            # Collective permute: send to next device in ring, receive from previous
            # This is equivalent to rotating features across devices
            modality2_features_permuted = self._collective_permute(modality2_features_permuted)

            # Compute loss for local images vs permuted texts
            # All pairs here are negatives (no positive pairs)
            logits_remote = torch.matmul(modality1_features, modality2_features_permuted.T) * temperature

            # All labels are -1 (all negative pairs)
            labels_remote = torch.full((local_batch_size, local_batch_size), -1.0,
                                       device=modality1_features.device)

            # Compute loss for this chunk
            loss_remote = -F.logsigmoid(labels_remote * (logits_remote + scaled_bias))
            total_loss += loss_remote.sum()
            num_loss_chunks += local_batch_size * local_batch_size

        # Average over all pairs
        # Total pairs = (world_size * local_batch_size)²
        global_batch_size = world_size * local_batch_size
        loss = total_loss / (global_batch_size * global_batch_size)

        return loss

    def _collective_permute(self, tensor):
        """
        Perform collective permute in a ring pattern.
        Each device sends its tensor to the next device and receives from previous.

        Ring pattern: 0 -> 1 -> 2 -> ... -> (D-1) -> 0

        Uses synchronous send/recv with optional barriers to avoid deadlocks.
        """
        if not dist.is_available() or not dist.is_initialized():
            return tensor

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        if world_size == 1:
            return tensor

        # Determine source and destination in the ring
        send_to = (rank + 1) % world_size
        recv_from = (rank - 1 + world_size) % world_size

        # Prepare buffer for receiving
        recv_tensor = torch.zeros_like(tensor)

        if self.ring_use_barrier:
            # Safer version with barriers (slower but avoids deadlocks)
            # Synchronize all processes before communication
            dist.barrier()

            # Split into even/odd ranks to avoid circular dependency
            if rank % 2 == 0:
                # Even ranks: send first, then receive
                dist.send(tensor.contiguous(), dst=send_to)
                dist.recv(recv_tensor, src=recv_from)
            else:
                # Odd ranks: receive first, then send
                dist.recv(recv_tensor, src=recv_from)
                dist.send(tensor.contiguous(), dst=send_to)

            # Synchronize after communication
            dist.barrier()
        else:
            # Faster version without barriers
            # Post receive first, then send
            recv_op = dist.irecv(recv_tensor, src=recv_from)
            send_op = dist.isend(tensor.contiguous(), dst=send_to)

            # Wait for both operations
            recv_op.wait()
            send_op.wait()

        return recv_tensor


class SimCLRLoss(nn.Module):
    """
    SimCLR NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss
    
    Used for self-supervised learning with augmented views of the same image.
    Each sample in the batch has a positive pair (different augmentation of same image)
    and 2N-2 negatives.
    """
    
    def __init__(self, temperature=0.07, learnable_temperature=False):
        """
        Args:
            temperature: temperature parameter for scaling logits (default: 0.07 from SimCLR paper)
            learnable_temperature: if True, temperature is a learnable parameter
        """
        super(SimCLRLoss, self).__init__()
        
        if learnable_temperature:
            # Store as log_temperature for numerical stability
            self.log_temperature = nn.Parameter(torch.ones([]) * np.log(temperature))
        else:
            self.register_buffer('log_temperature', torch.tensor(np.log(temperature)))
    
    def forward(self, z_i, z_j):
        """
        Compute NT-Xent loss for a batch of augmented pairs.
        
        Args:
            z_i: embeddings from first augmentation [batch_size, feature_dim]
            z_j: embeddings from second augmentation [batch_size, feature_dim]
        
        Returns:
            NT-Xent loss
        """
        batch_size = z_i.shape[0]
        
        z_i = F.normalize(z_i, dim=-1)
        z_j = F.normalize(z_j, dim=-1)
        
        representations = torch.cat([z_i, z_j], dim=0)
        
        temperature = self.log_temperature.exp()
        similarity_matrix = torch.matmul(representations, representations.T) / temperature
        
        # Create masks for positive and negative pairs
        # Positive pairs: (i, i+batch_size) and (i+batch_size, i)
        pos_mask = torch.zeros(2 * batch_size, 2 * batch_size, dtype=torch.bool, device=z_i.device)
        pos_mask[torch.arange(batch_size), batch_size + torch.arange(batch_size)] = True
        pos_mask[batch_size + torch.arange(batch_size), torch.arange(batch_size)] = True
        
        # Negative mask: all except self-similarity and positive pairs
        neg_mask = torch.ones(2 * batch_size, 2 * batch_size, dtype=torch.bool, device=z_i.device)
        neg_mask[torch.arange(2 * batch_size), torch.arange(2 * batch_size)] = False  # Remove self
        neg_mask = neg_mask & ~pos_mask  # Remove positive pairs
        
        # Extract positive similarities
        pos_sim = similarity_matrix[pos_mask].view(2 * batch_size, 1)
        
        # Extract negative similarities
        neg_sim = similarity_matrix[neg_mask].view(2 * batch_size, -1)
        
        # Concatenate positive and negatives: [2*batch_size, 1 + 2(N-1)]
        logits = torch.cat([pos_sim, neg_sim], dim=1)
        
        # Labels: positive is always at index 0
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z_i.device)
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        
        return loss


def get_contrastive_loss(loss_type, **kwargs):
    """
    Factory function to get contrastive loss by name

    Args:
        loss_type: name of loss function
        **kwargs: arguments for the loss function

    Returns:
        loss function instance

    Example:
        >>> loss_fn = get_contrastive_loss('simclr', temperature=0.5)
        >>> loss = loss_fn(z_i, z_j)
    """
    loss_type = loss_type.lower()

    if loss_type == 'clip':
        return CLIPLoss(**kwargs)
    elif loss_type == 'siglip':
        return SigLIPLoss(**kwargs)
    elif loss_type == 'simclr' or loss_type == 'ntxent':
        return SimCLRLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Choose 'clip', 'siglip', or 'simclr'.")
