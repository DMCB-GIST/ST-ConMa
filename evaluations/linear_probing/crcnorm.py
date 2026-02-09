import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import json
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from sklearn.metrics import f1_score, balanced_accuracy_score
from torchvision import transforms

sys.path.append("./utils")
from pt_load_inference import VisionInference

# Model configuration
VISION_MODEL_NAME = "pathoduet"  # Options: 'pathoduet', 'phikon', 'uni2_h'
CHECKPOINT_PATH = "./checkpoints/st_conma_pythia410m_12layers_3aug_clip/checkpoint_epoch_12.pt"

MODEL_NAME = "ST_ConMa_text"
BASE_DIR = "./results/linear_probing/crcnorm/st_conma/"
BASE_PATH = f"{BASE_DIR}"
SEEDS = [42, 123, 456, 789, 101]

LRs = [0.001, 0.0001]
WDs = [0.1, 0.01, 0.001, 0.0001]
MAX_EPOCHS = 10000
K_FOLDS = 3
BATCH_SIZE = 32
INITIAL_SEED = 42

os.makedirs(BASE_PATH, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False 

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_transform(vision_model):
    """
    Use the transform from the vision model to ensure consistency

    Args:
        vision_model: VisionInference model with its own transform

    Returns:
        img_transform: transform from the vision model
    """
    return vision_model.transform

def load_crc_dataset():
    train_folder_path = "./ft_dataset/linear_probing/crc/NCT-CRC-HE-100K"
    test_folder_path = "./ft_dataset/linear_probing/crc/CRC-VAL-HE-7K"
    train_data = []
    test_data = []
    for root, _, files in os.walk(train_folder_path):
        for file in sorted(files):
            if file.lower().endswith(".tif"):
                train_data.append(os.path.join(root, file))  
    for root, _, files in os.walk(test_folder_path):
        for file in sorted(files):
            if file.lower().endswith(".tif"):
                test_data.append(os.path.join(root, file))
    train_data = sorted(train_data)
    test_data = sorted(test_data)
    return train_data, test_data

class CRCDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform
        self.label_map = self._get_labels() 

    def _get_labels(self):
        labels = {}
        class_names = sorted(set([os.path.basename(os.path.dirname(path)) for path in self.file_paths]))
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
        for path in self.file_paths:
            label = os.path.basename(os.path.dirname(path)) 
            labels[path] = self.class_to_idx[label] 
        return labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image_path = self.file_paths[idx]
        image = Image.open(image_path)
        label = self.label_map[image_path]  
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)
    
    def get_num_classes(self):
        return len(self.class_to_idx)

def extract_features(vision_model, dataloader):
    """
    Extract vision embeddings from ST-ConMa-text vision encoder

    Args:
        vision_model: VisionInference model with pretrained weights
        dataloader: DataLoader providing images

    Returns:
        features: [N, vision_dim] tensor
        labels: [N] tensor
    """
    vision_model.model.eval()
    features, labels = [], []

    with torch.no_grad():
        for images, target in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)

            # Extract vision embeddings
            # model returns (patch_features, representation)
            _, image_features = vision_model.model(images)  # [B, vision_dim]

            image_features = image_features.cpu()
            features.append(image_features)
            labels.append(target)

    features = torch.cat(features, dim=0)
    labels = torch.cat([l.view(-1) for l in labels], dim=0)
    return features, labels

class LinearProbe(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        return self.linear(x)

def train_model(model, train_loader, val_loader, learning_rate, weight_decay, num_epochs, device, patience=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_val_loss = float('inf')
    best_epoch = 0
    best_model_state = None
    patience_counter = 0  
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() 
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels_list = []
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item() 
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())
        val_loss /= len(val_loader)
        val_f1 = f1_score(val_labels_list, val_preds, average="macro")
        
        # Removed detailed epoch output to reduce clutter
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                # Removed early stopping print
                break
    model.load_state_dict(best_model_state)
    return model, best_epoch, best_val_loss

def cross_validate(features, labels, input_dim, output_dim, learning_rate, weight_decay, max_epochs, kfold_splits, seed):
    # Use the specific splits for this seed
    split_indices = []
    for fold in sorted(kfold_splits.keys(), key=int):
        train_idx = np.array(kfold_splits[fold]["train_idx"])
        val_idx = np.array(kfold_splits[fold]["val_idx"])
        split_indices.append((train_idx, val_idx))
      
    # Set seed for reproducibility within cross validation
    seed_everything(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    
    fold_scores = []
    best_epochs = []
    for fold, (train_idx, val_idx) in enumerate(split_indices):
        print(f"Fold {fold+1}/{K_FOLDS}")
        
        train_idx_tensor = torch.tensor(train_idx, dtype=torch.long)
        val_idx_tensor = torch.tensor(val_idx, dtype=torch.long)
        
        train_features = features.index_select(0, train_idx_tensor)
        train_labels = labels.index_select(0, train_idx_tensor)
        val_features = features.index_select(0, val_idx_tensor)
        val_labels = labels.index_select(0, val_idx_tensor)
        
        train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
        val_dataset = torch.utils.data.TensorDataset(val_features, val_labels)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=g
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=False,
            worker_init_fn=seed_worker,
            generator=g
        )
        
        model = LinearProbe(input_dim, output_dim)
        model.to(device)
        
        model, best_epoch, val_loss = train_model(model, train_loader, val_loader, learning_rate, weight_decay, MAX_EPOCHS, device)
        model.eval()
        val_preds = []
        val_labels_list = []
        with torch.no_grad():
            for features_batch, labels_batch in val_loader:
                features_batch, labels_batch = features_batch.to(device), labels_batch.to(device)
                outputs = model(features_batch)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels_list.extend(labels_batch.cpu().numpy())
        val_f1 = f1_score(val_labels_list, val_preds, average="macro")
        fold_scores.append(val_f1)
        best_epochs.append(best_epoch)
        print(f"Fold {fold+1}: best epoch = {best_epoch}, F1 = {val_f1:.4f}")
    
    avg_score = np.mean(fold_scores)
    avg_epoch = int(np.mean(best_epochs))
    return avg_score, avg_epoch, fold_scores

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    f1 = f1_score(all_labels, all_preds, average="macro")
    bacc = balanced_accuracy_score(all_labels, all_preds)
    return f1, bacc

def get_kfold_indices(num_samples, k_folds, seed):
    indices = np.arange(num_samples)
    np.random.seed(seed)  # Set seed specifically for the fold generation
    np.random.shuffle(indices)
    
    fold_splits = {}
    fold_size = num_samples // k_folds
    
    for fold in range(k_folds):
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size if fold < k_folds - 1 else num_samples
        
        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
        
        fold_splits[str(fold)] = {
            "train_idx": train_idx.tolist(),
            "val_idx": val_idx.tolist()
        }
    
    return fold_splits

def run_experiment(seed, train_features, train_labels, test_features, test_labels, num_classes, num_samples):
    print(f"\n================ Running experiment with seed {seed} ================")
    
    # Create or load the k-fold splits for this specific seed
    splits_dir = f"{BASE_DIR}kfold_splits"
    os.makedirs(splits_dir, exist_ok=True)
    splits_save_path = f"{splits_dir}/kfold_splits_seed_{seed}.json"
    
    if os.path.exists(splits_save_path):
        print(f"Loading existing KFold splits for seed {seed}")
        with open(splits_save_path, "r") as f:
            saved_splits = json.load(f)
        
        max_index = max([max(fold["train_idx"]) for fold in saved_splits.values()])
        if max_index >= num_samples:
            raise ValueError(f"Saved KFold splits for seed {seed} don't match current dataset size")
    else:
        print(f"Creating new KFold splits for seed {seed}")
        saved_splits = get_kfold_indices(num_samples, K_FOLDS, seed)
        with open(splits_save_path, "w") as f:
            json.dump(saved_splits, f, indent=4)
        print(f"KFold splits saved to: {splits_save_path}")
    
    input_dim = train_features.shape[1]
    output_dim = num_classes
    
    print(f"Running hyperparameter search for seed {seed}...")
    best_score = 0
    best_params = None
    best_epoch = 0
    best_fold_scores = None
    
    results = []
    
    # Run hyperparameter search with cross-validation
    for lr in LRs:
        for wd in WDs:
            print(f"\nTesting: LR={lr}, Weight Decay={wd}")
            
            avg_score, avg_epoch, fold_scores = cross_validate(
                train_features, train_labels, input_dim, output_dim,
                lr, wd, MAX_EPOCHS, saved_splits, seed
            )
            
            results.append({
                "lr": lr,
                "weight_decay": wd,
                "avg_epoch": avg_epoch,
                "avg_f1": avg_score,
                "fold_scores": fold_scores
            })
            
            print(f"Average F1: {avg_score:.4f}, Average Best Epoch: {avg_epoch}")
            
            if avg_score > best_score:
                best_score = avg_score
                best_params = (lr, wd)
                best_epoch = avg_epoch
                best_fold_scores = fold_scores
    
    print(f"\nBest hyperparameters for seed {seed}:")
    print(f"LR={best_params[0]}, Weight Decay={best_params[1]}, Epochs={best_epoch}")
    print(f"Cross-validation F1: {best_score:.4f}")
    
    # Set seed for final training
    seed_everything(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    
    # Train final model on all training data with best hyperparameters
    print("\nTraining final model with best hyperparameters...")
    final_train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    final_test_dataset = torch.utils.data.TensorDataset(test_features, test_labels)
    
    final_train_loader = DataLoader(
        final_train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g
    )
    final_test_loader = DataLoader(
        final_test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=g
    )
    
    final_model = LinearProbe(input_dim, output_dim)
    final_optimizer = Adam(final_model.parameters(), lr=best_params[0], weight_decay=best_params[1])
    criterion = nn.CrossEntropyLoss()
    
    final_model.to(device)
    
    # Train final model
    for epoch in range(best_epoch):
        final_model.train()
        epoch_loss = 0.0
        for features, labels in final_train_loader:
            features, labels = features.to(device), labels.to(device)
            final_optimizer.zero_grad()
            outputs = final_model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            final_optimizer.step()
            epoch_loss += loss.item()
        
        # Simplified epoch progress reporting
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == best_epoch - 1:
            print(f"Epoch {epoch+1}/{best_epoch}")

    # Evaluate on test set
    test_f1, test_bacc = evaluate_model(final_model, final_test_loader, device)
    
    print(f"\nTest Results for seed {seed}:")
    print(f"F1 Score: {test_f1:.4f}")
    print(f"Balanced Accuracy: {test_bacc:.4f}")
    
    return {
        "seed": seed,
        "best_hyperparameters": {
            "lr": best_params[0],
            "weight_decay": best_params[1],
            "epochs": best_epoch
        },
        "cross_validation": {
            "avg_f1": best_score,
            "fold_scores": best_fold_scores
        },
        "test_f1": test_f1,
        "test_balanced_accuracy": test_bacc,
        "all_hyperparameter_results": results,
        "kfold_splits_path": splits_save_path
    }

def main():
    print(f"Running experiments with {len(SEEDS)} different random seeds")
    all_results = []

    # Lists to store metrics for statistical analysis
    f1_scores = []
    bacc_scores = []

    # Use initial seed for model loading and feature extraction
    seed_everything(INITIAL_SEED)
    g = torch.Generator()
    g.manual_seed(INITIAL_SEED)

    print("Loading ST-ConMa-text vision encoder...")
    print(f"  Vision model: {VISION_MODEL_NAME}")
    print(f"  Checkpoint: {CHECKPOINT_PATH}")

    # Load pretrained ST-ConMa-text vision encoder
    vision_model = VisionInference(
        checkpoint_path=CHECKPOINT_PATH,
        model_name=VISION_MODEL_NAME,
        device=device
    )

    print("Loading CRC dataset (normalized version)...")
    train_files, test_files = load_crc_dataset()

    # Get transform from vision model to ensure consistency
    crc_img_transform = set_transform(vision_model)

    train_dataset = CRCDataset(file_paths=train_files, transform=crc_img_transform)
    test_dataset = CRCDataset(file_paths=test_files, transform=crc_img_transform)
    num_classes = train_dataset.get_num_classes()
    print(f"CRC dataset has {num_classes} classes")
    
    # Create data loaders with the initial seed
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=8, 
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=g
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=8, 
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=g
    )
    
    # Extract features once for all seeds
    print("Extracting features... (only once for all seeds)")
    train_features, train_labels = extract_features(vision_model, train_dataloader)
    test_features, test_labels = extract_features(vision_model, test_dataloader)
    
    num_samples = train_features.shape[0]
    print(f"Train features shape: {train_features.shape}, Train labels shape: {train_labels.shape}")
    print(f"Test features shape: {test_features.shape}, Test labels shape: {test_labels.shape}")
    
    # Create splits directory to store all seed-specific splits
    splits_dir = f"{BASE_DIR}kfold_splits"
    os.makedirs(splits_dir, exist_ok=True)
    
    # Run experiments for each seed
    for seed in SEEDS:
        result = run_experiment(
            seed, 
            train_features, 
            train_labels, 
            test_features, 
            test_labels, 
            num_classes,
            num_samples
        )
        all_results.append(result)
        f1_scores.append(result["test_f1"])
        bacc_scores.append(result["test_balanced_accuracy"])
    
    # Calculate mean and standard deviation
    f1_mean = np.mean(f1_scores)
    f1_std = np.std(f1_scores)
    bacc_mean = np.mean(bacc_scores)
    bacc_std = np.std(bacc_scores)
    
    # Print individual results
    print("\n===== Individual Test Results =====")
    for idx, result in enumerate(all_results):
        print(f"Seed {result['seed']}: F1 = {result['test_f1']:.4f}, BACC = {result['test_balanced_accuracy']:.4f}")
    
    # Print summary statistics
    print("\n===== Summary Statistics =====")
    print(f"F1 Score: {f1_mean:.4f} ± {f1_std:.4f}")
    print(f"Balanced Accuracy: {bacc_mean:.4f} ± {bacc_std:.4f}")
    
    # Save all results to JSON
    final_results = {
        "individual_results": all_results,
        "summary": {
            "f1_score": {
                "mean": float(f1_mean),
                "std": float(f1_std),
                "values": [float(x) for x in f1_scores]
            },
            "balanced_accuracy": {
                "mean": float(bacc_mean),
                "std": float(bacc_std),
                "values": [float(x) for x in bacc_scores]
            }
        },
        "seeds_used": SEEDS,
        "kfold_splits_dir": splits_dir
    }
    
    results_path = f"{BASE_PATH}/crcnorm_linear_probing_multi_seed_results.json"
    
    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=4)
    
    print(f"Results saved to {results_path}")
    print(f"K-fold splits for all seeds saved to {splits_dir}")

if __name__ == "__main__":
    main()