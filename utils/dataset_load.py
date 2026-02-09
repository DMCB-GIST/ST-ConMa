import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from torch.utils.data import Dataset, Sampler
from collections import defaultdict
import random
from PIL import Image
import pandas as pd
from typing import Dict, Optional, Tuple
import numpy as np
from pathlib import Path


class VisionGeneDataset(Dataset):
    def __init__(self, image_dir, gene_file, transform=None, tokenizer=None,
                 image_extension='.png', except_ft_samples=True, augmentation=None,
                 use_multitask=True):
        self.image_dir = image_dir
        self.transform = transform
        self.augmentation = augmentation
        self.tokenizer = tokenizer
        self.image_extension = image_extension
        self.use_multitask = use_multitask  # For multi-task learning (IGC, IIC, IGM)

        self.gene_data = pd.read_csv(gene_file, dtype={'id': str, 'sentence': str})

        required_columns = ['id', 'sentence']
        for col in required_columns:
            if col not in self.gene_data.columns:
                raise ValueError(f"Missing required column: {col}")

        if except_ft_samples:
            except_HLT = {f"NCBI{i}" for i in range(672, 676)}
            except_HER2ST = {f"SPA{i}" for i in range(119, 155)}
            except_SCC    = {f"NCBI{i}" for i in range(759, 771)}
            except_samples = except_HER2ST | except_SCC | except_HLT
        else:
            except_samples = set()  

        img_dir = Path(self.image_dir)
        present_ids = {
            p.stem  
            for p in img_dir.iterdir()
            if p.is_file() and p.suffix == self.image_extension
        }

        df = self.gene_data

        sample_prefix = df['id'].str.split('_', n=1, expand=True)[0] 

        mask_exists = df['id'].isin(present_ids)
        mask_except = ~sample_prefix.isin(except_samples)

        self.valid_samples = df.index[mask_exists & mask_except].tolist()

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx) -> Dict:
        # Get actual data index
        data_idx = self.valid_samples[idx]
        row = self.gene_data.iloc[data_idx]

        # Load image
        image_path = os.path.join(
            self.image_dir,
            f"{row['id']}{self.image_extension}"
        )
        image = Image.open(image_path).convert('RGB')

        # For multi-task learning, create 3 augmented versions
        if self.use_multitask:
            # Check if augmentation is a tuple of (aug1, aug2, aug3)
            if self.augmentation and isinstance(self.augmentation, tuple) and len(self.augmentation) == 3:
                aug1_transform, aug2_transform, aug3_transform = self.augmentation
                # Apply each augmentation separately
                image_aug1 = aug1_transform(image) 
                image_aug2 = aug2_transform(image)  
                image_aug3 = aug3_transform(image) 
            elif self.augmentation:
                # Fallback: use single augmentation for all
                image_aug1 = self.augmentation(image)
                image_aug2 = self.augmentation(image)
                image_aug3 = self.augmentation(image)
            else:
                # No augmentation, use transform only
                if self.transform:
                    image_aug1 = self.transform(image)
                    image_aug2 = self.transform(image)
                    image_aug3 = self.transform(image)
                else:
                    # This shouldn't happen in practice, but handle it
                    raise ValueError("use_multitask=True requires either augmentation or transform")
        else:
            if self.augmentation:
                image = self.augmentation(image)
            if self.transform:
                image = self.transform(image)
            image_aug1 = None
            image_aug2 = None
            image_aug3 = None

        # Tokenize st sentence or load pre-tokenized data
        st_sentence = row['sentence']

        result = {
            'image': image,
            'sentence': st_sentence,
            'id': row['id']
        }

        # Add augmented images for multi-task
        if self.use_multitask:
            result['image_aug1'] = image_aug1
            result['image_aug2'] = image_aug2
            result['image_aug3'] = image_aug3

        return result
    

def collate_fn(batch, tokenizer=None, max_length=512, use_multitask=True):
    # For multi-task learning
    if use_multitask:
        # Stack 3 augmented images
        images_aug1 = torch.stack([b['image_aug1'] for b in batch])  # Unaugmented view for IGC
        images_aug2 = torch.stack([b['image_aug2'] for b in batch])  # Augmented view 1 for IIC
        images_aug3 = torch.stack([b['image_aug3'] for b in batch])  # Augmented view 2 for IIC

        # Get gene sentences directly (already space-separated in CSV)
        gene_sentences = [b['sentence'] for b in batch]

        ids = [b['id'] for b in batch]
        return {
            'images_aug1': images_aug1,
            'images_aug2': images_aug2,
            'images_aug3': images_aug3,
            'gene_sentences': gene_sentences,  # Space-separated gene names
            'ids': ids
        }

    images = torch.stack([b['image'] for b in batch])

    sentences = [b['sentence'] for b in batch]
    if tokenizer is not None:
        toks = tokenizer(
            sentences,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=max_length
        )
        tokens = {'input_ids': toks['input_ids'], 'attention_mask': toks['attention_mask']}
    else:
        tokens = {'sentence': sentences}

    ids = [b['id'] for b in batch]
    return {'images': images, 'tokens': tokens, 'ids': ids}


def create_dataloaders(
    train_image_dir: str,
    train_gene_file: str,
    transform=None,
    tokenizer=None,
    batch_size: int = 32,
    num_workers: int = 4,
    prefetch_factor: int = 4,
    use_ddp: bool = True,
    except_ft_samples: bool = True,
    augmentation=None,
    use_multitask: bool = False
):
    """
    Create train and validation dataloaders.

    Args:
        train_image_dir: training images directory
        train_gene_file: training gene file
        transform: image transformation
        tokenizer: tokenizer 
        batch_size: batch size
        num_workers: number of workers
        use_ddp: whether using DDP
        except_ft_samples: whether to exclude fine-tuning samples
        augmentation: optional augmentation transform
        use_multitask: whether to use multi-task learning (IGC, IIC, IGM)

    Returns:
        train_loader
    """
    # Create datasets
    train_dataset = VisionGeneDataset(
        image_dir=train_image_dir,
        gene_file=train_gene_file,
        transform=transform,
        tokenizer=tokenizer,
        except_ft_samples=except_ft_samples,
        augmentation=augmentation,
        use_multitask=use_multitask
    )

    # Create samplers for DDP
    if use_ddp:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
    else:
        train_sampler = None

    # Create train loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler, # DDP sampler (splits data among GPUs)
        num_workers=num_workers,
        prefetch_factor=prefetch_factor, # Number of batches preloaded per worker
        collate_fn=lambda batch: collate_fn(
            batch,
            tokenizer=tokenizer,
            max_length=512,
            use_multitask=use_multitask
        ),
        pin_memory=True, 
        drop_last=True
    )

    return train_loader