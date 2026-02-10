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
from datasets import load_dataset
torch.multiprocessing.set_sharing_strategy('file_system')

sys.path.append("./utils")
from pt_load_inference import VisionInference

# Model configuration
VISION_MODEL_NAME = "pathoduet" 
CHECKPOINT_PATH = "./checkpoints/st_conma_pythia410m_12layers_3aug_clip/checkpoint_epoch_12.pt"

MODEL_NAME = "ST_ConMa"
BASE_DIR = "./results/linear_probing/cam17/st_conma/"
BASE_PATH = f"{BASE_DIR}"
SEEDS = [42, 123, 456, 789, 101]

LRs = [0.001, 0.0001]
WDs = [0.1, 0.01, 0.001, 0.0001]
MAX_EPOCHS = 10000
BATCH_SIZE = 32

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

def load_cam17_dataset():
    dataset = load_dataset("wltjr1007/Camelyon17-WILDS")
    train_data = dataset["train"]
    valid_data = dataset["validation"]
    test_data = dataset["test"]
    
    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(valid_data)}")
    print(f"Test samples: {len(test_data)}")
    
    return train_data, valid_data, test_data

class CAM17Dataset(Dataset):
    def __init__(self, data, transforms=None):
        self.data = data
        self.img_transform = transforms

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx]["image"]
        label = torch.tensor(self.data[idx]["label"], dtype=torch.long)

        # Convert RGBA to RGB if necessary
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        if self.img_transform:
            transformed_img = self.img_transform(image)
        else:
            raise ValueError("Error: img_transform is None. Please provide a valid transformation.")

        return transformed_img, label

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
        
        # Weight initialization
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        # L2 normalization
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
        
        train_loss = train_loss / len(train_loader)
        
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
        
        val_loss = val_loss / len(val_loader)
        val_f1 = f1_score(val_labels_list, val_preds, average="macro")
        
        # Removed validation output print to reduce unnecessary output
        
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

def run_experiment(seed, train_features, train_labels, valid_features, valid_labels, test_features, test_labels):
    # Set random seed for reproducibility
    seed_everything(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    
    print(f"\n================ Running experiment with seed {seed} ================")
    
    input_dim = train_features.shape[1]
    output_dim = 2  # num_classes
    
    print("Finding best hyperparameters...")
    best_score = 0
    best_params = None
    best_epoch = 0
    
    # Quick hyperparameter search - minimal output
    for lr in LRs:
        for wd in WDs:            
            train_tensor_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
            valid_tensor_dataset = torch.utils.data.TensorDataset(valid_features, valid_labels)
            
            train_loader = DataLoader(
                train_tensor_dataset, 
                batch_size=BATCH_SIZE, 
                shuffle=True,
                worker_init_fn=seed_worker,
                generator=g
            )
            valid_loader = DataLoader(
                valid_tensor_dataset, 
                batch_size=BATCH_SIZE, 
                shuffle=False,
                worker_init_fn=seed_worker,
                generator=g
            )
            
            probe_model = LinearProbe(input_dim, output_dim)
            probe_model = probe_model.to(device)
            
            probe_model, best_epoch_val, val_loss = train_model(
                probe_model, train_loader, valid_loader, lr, wd, MAX_EPOCHS, device
            )
            
            probe_model.eval()
            val_preds = []
            val_labels_list = []
            
            with torch.no_grad():
                for features, labels in valid_loader:
                    features, labels = features.to(device), labels.to(device)
                    outputs = probe_model(features)
                    _, preds = torch.max(outputs, 1)
                    val_preds.extend(preds.cpu().numpy())
                    val_labels_list.extend(labels.cpu().numpy())
            
            val_f1 = f1_score(val_labels_list, val_preds, average="macro")
            
            if val_f1 > best_score:
                best_score = val_f1
                best_params = (lr, wd)
                best_epoch = best_epoch_val
    
    print(f"Best hyperparameters: LR={best_params[0]}, WD={best_params[1]}, Epochs={best_epoch}")
    
    print("\nTraining final model with best hyperparameters...")
    
    train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_features, test_labels)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=g
    )
    
    final_model = LinearProbe(input_dim, output_dim)
    final_optimizer = Adam(final_model.parameters(), lr=best_params[0], weight_decay=best_params[1])
    criterion = nn.CrossEntropyLoss()
    
    final_model.to(device)
    
    for epoch in range(best_epoch):
        final_model.train()
        epoch_loss = 0.0
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            final_optimizer.zero_grad()
            outputs = final_model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            final_optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        # Removed epoch progress print

    test_f1, test_bacc = evaluate_model(final_model, test_loader, device)
    
    print(f"Test Results with seed {seed}:")
    print(f"F1 Score: {test_f1:.4f}")
    print(f"Balanced Accuracy: {test_bacc:.4f}")
    
    return {
        "seed": seed,
        "best_hyperparameters": {
            "lr": best_params[0],
            "weight_decay": best_params[1],
            "epochs": best_epoch
        },
        "test_f1": test_f1,
        "test_balanced_accuracy": test_bacc
    }

def main():
    print(f"Running experiments with {len(SEEDS)} different random seeds")
    all_results = []

    # Lists to store metrics for statistical analysis
    f1_scores = []
    bacc_scores = []

    print("Loading ST-ConMa-text vision encoder...")
    print(f"  Vision model: {VISION_MODEL_NAME}")
    print(f"  Checkpoint: {CHECKPOINT_PATH}")

    # Load pretrained ST-ConMa-text vision encoder
    vision_model = VisionInference(
        checkpoint_path=CHECKPOINT_PATH,
        model_name=VISION_MODEL_NAME,
        device=device
    )

    print("Loading CAM17 dataset...")
    train_data, valid_data, test_data = load_cam17_dataset()

    # Get transform from vision model to ensure consistency
    cam17_img_transform = set_transform(vision_model)

    # Use initial seed for dataset loading and feature extraction
    seed_everything(SEEDS[0])
    g = torch.Generator()
    g.manual_seed(SEEDS[0])

    train_dataset = CAM17Dataset(data=train_data, transforms=cam17_img_transform)
    valid_dataset = CAM17Dataset(data=valid_data, transforms=cam17_img_transform)
    test_dataset = CAM17Dataset(data=test_data, transforms=cam17_img_transform)
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=8, 
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=g
    )
    
    valid_dataloader = DataLoader(
        valid_dataset, 
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
    
    print("Extracting features... (only once for all seeds)")
    train_features, train_labels = extract_features(vision_model, train_dataloader)
    valid_features, valid_labels = extract_features(vision_model, valid_dataloader)
    test_features, test_labels = extract_features(vision_model, test_dataloader)
    
    # Now run experiments with different seeds for classifier training
    for seed in SEEDS:
        result = run_experiment(seed, train_features, train_labels, valid_features, valid_labels, test_features, test_labels)
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
        }
    }
    
    results_path = f"{BASE_PATH}/cam17_linear_probing_multi_seed_results.json"
    
    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=4)
    
    print(f"Results saved to {results_path}")

if __name__ == "__main__":
    main()
