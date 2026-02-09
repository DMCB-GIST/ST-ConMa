import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import re
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
VISION_MODEL_NAME = "pathoduet"  
CHECKPOINT_PATH = "./checkpoints/st_conma_pythia410m_12layers_3aug_clip/checkpoint_epoch_12.pt"

MODEL_NAME = "ST_ConMa_text"
BASE_DIR = "./results/linear_probing/luad/st_conma/"
BASE_PATH = f"{BASE_DIR}"

LRs = [0.001, 0.0001]
WDs = [0.1, 0.01, 0.001, 0.0001]
MAX_EPOCHS = 10000
BATCH_SIZE = 32
SEED = 42
N_FOLDS = 5

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

seed_everything(SEED)
g = torch.Generator()
g.manual_seed(SEED)

def set_transform(vision_model):
    """
    Use the transform from the vision model to ensure consistency

    Args:
        vision_model: VisionInference model with its own transform

    Returns:
        img_transform: transform from the vision model
    """
    return vision_model.transform

def load_luad_dataset():
    dataset_path = "./ft_dataset/linear_probing/luad/"
    
    file_list = []
    for root, _, files in os.walk(dataset_path):  
        # Sort files to ensure consistent order
        for file in sorted(files):
            if file.endswith('.png'):
                file_list.append(os.path.join(root, file))

    data = []
    for file in file_list:
        filename = os.path.basename(file)
        # Extract SampleID and label info using regex
        match = re.match(r"(.+)-[\d]+-[\d]+-\[(\d), (\d), (\d)\]\.png", filename)  
        if match:
            sample_id, label1, label2, label3 = match.groups()
            label = f"{label1},{label2},{label3}"
            # Only select files with exactly one '1' among the three labels
            if (label1, label2, label3).count('1') == 1:  
                data.append((sample_id, file, label))

    df = pd.DataFrame(data, columns=["SampleID", "Filename", "Label"])
    df = df.sort_values("SampleID").reset_index(drop=True)
    
    # Add label class (0, 1, or 2 based on which position has a 1)
    df["LabelClass"] = df["Label"].apply(lambda x: x.split(",").index("1"))
    
    print(f"Total samples (patients): {len(df['SampleID'].unique())}")
    print(f"Total images: {len(df)}")
    print(f"Class distribution: {df['LabelClass'].value_counts().to_dict()}")
    
    return df

class LUADDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.img_transform = transforms

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        image_path = self.df.iloc[idx]["Filename"]
        image = Image.open(image_path).convert("RGB")
        label = self.df.iloc[idx]["LabelClass"]

        if self.img_transform:
            transformed_img = self.img_transform(image)
        else:
            raise ValueError("Error: img_transform is None. Please provide a valid transformation.")

        return transformed_img, torch.tensor(label, dtype=torch.long)

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
                # Early stopping
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

def get_patient_wise_folds(df, n_folds=5, seed=42):
    # Get unique patient IDs
    patient_ids = df['SampleID'].unique()
    
    # Create GroupKFold for patient-wise splitting
    np.random.seed(seed)
    np.random.shuffle(patient_ids)
    
    fold_size = len(patient_ids) // n_folds
    folds = {}
    
    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size if fold < n_folds - 1 else len(patient_ids)
        
        val_patients = patient_ids[val_start:val_end]
        train_patients = np.array([pid for pid in patient_ids if pid not in val_patients])
        
        # Get indices of samples belonging to train/val patients
        train_indices = df[df['SampleID'].isin(train_patients)].index.tolist()
        val_indices = df[df['SampleID'].isin(val_patients)].index.tolist()
        
        folds[str(fold)] = {
            "train_patients": train_patients.tolist(),
            "val_patients": val_patients.tolist(),
            "train_indices": train_indices,
            "val_indices": val_indices
        }
    
    return folds

def run_fold_experiment(fold_idx, fold_data, features, labels, input_dim, output_dim):
    print(f"\n================ Running Fold {fold_idx+1}/{N_FOLDS} ================")
    
    # Get train and validation indices for this fold
    train_indices = fold_data["train_indices"]
    val_indices = fold_data["val_indices"]
    
    # Convert to tensor indices
    train_idx_tensor = torch.tensor(train_indices, dtype=torch.long)
    val_idx_tensor = torch.tensor(val_indices, dtype=torch.long)
    
    # Get train and validation features and labels
    train_features = features.index_select(0, train_idx_tensor)
    train_labels = labels.index_select(0, train_idx_tensor)
    val_features = features.index_select(0, val_idx_tensor)
    val_labels = labels.index_select(0, val_idx_tensor)
    
    # Find best hyperparameters for this fold
    best_score = 0
    best_params = None
    best_epoch = 0
    
    for lr in LRs:
        for wd in WDs:
            print(f"\nTesting: LR={lr}, Weight Decay={wd}")
            
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
            
            model, best_epoch_val, val_loss = train_model(
                model, train_loader, val_loader, lr, wd, MAX_EPOCHS, device
            )
            
            # Evaluate on validation set
            val_f1, val_bacc = evaluate_model(model, val_loader, device)
            print(f"LR={lr}, WD={wd}, Epochs={best_epoch_val}, Val F1={val_f1:.4f}, Val BACC={val_bacc:.4f}")
            
            if val_f1 > best_score:
                best_score = val_f1
                best_params = (lr, wd)
                best_epoch = best_epoch_val
    
    print(f"\nBest hyperparameters for fold {fold_idx+1}:")
    print(f"LR={best_params[0]}, Weight Decay={best_params[1]}, Epochs={best_epoch}")
    print(f"Validation F1: {best_score:.4f}")
    
    # Train final model for this fold with best hyperparameters
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
    
    final_model = LinearProbe(input_dim, output_dim)
    final_optimizer = Adam(final_model.parameters(), lr=best_params[0], weight_decay=best_params[1])
    criterion = nn.CrossEntropyLoss()
    
    final_model.to(device)
    
    # Train final model with best hyperparameters
    for epoch in range(best_epoch):
        final_model.train()
        epoch_loss = 0.0
        
        for features_batch, labels_batch in train_loader:
            features_batch, labels_batch = features_batch.to(device), labels_batch.to(device)
            final_optimizer.zero_grad()
            outputs = final_model(features_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            final_optimizer.step()
            
            epoch_loss += loss.item() 
        
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == best_epoch - 1:
            print(f"Epoch {epoch+1}/{best_epoch}")
    
    fold_f1, fold_bacc = evaluate_model(final_model, val_loader, device)
    
    print(f"\nFold {fold_idx+1} Results:")
    print(f"F1 Score: {fold_f1:.4f}")
    print(f"Balanced Accuracy: {fold_bacc:.4f}")
    
    fold_results = {
        "fold": fold_idx + 1,
        "train_patients": len(fold_data["train_patients"]),
        "val_patients": len(fold_data["val_patients"]),
        "train_samples": len(train_indices),
        "val_samples": len(val_indices),
        "best_hyperparameters": {
            "lr": best_params[0],
            "weight_decay": best_params[1],
            "epochs": best_epoch
        },
        "validation_f1": fold_f1,
        "validation_bacc": fold_bacc
    }
    
    return fold_results, fold_f1, fold_bacc

def main():

    print("Loading ST-ConMa-text vision encoder...")
    print(f"  Vision model: {VISION_MODEL_NAME}")
    print(f"  Checkpoint: {CHECKPOINT_PATH}")

    # Load pretrained ST-ConMa-text vision encoder
    vision_model = VisionInference(
        checkpoint_path=CHECKPOINT_PATH,
        model_name=VISION_MODEL_NAME,
        device=device
    )

    print("Loading LUAD dataset...")
    df = load_luad_dataset()

    # Get transform from vision model to ensure consistency
    luad_img_transform = set_transform(vision_model)

    # Create dataset with all samples
    dataset = LUADDataset(df=df, transforms=luad_img_transform)
    
    num_classes = 3  # LUAD dataset has 3 classes (position of '1' in the label)
    print(f"Number of classes: {num_classes}")
    
    # Create dataloader for feature extraction
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=8, 
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=g
    )
    
    print("Extracting features...")
    features, labels = extract_features(vision_model, dataloader)
    
    input_dim = features.shape[1]
    output_dim = num_classes
    
    print(f"Features shape: {features.shape}")
    print(f"Input dimension: {input_dim}")
    
    folds_dir = f"{BASE_DIR}patient_wise_folds"
    os.makedirs(folds_dir, exist_ok=True)
    folds_path = f"{folds_dir}/patient_wise_{N_FOLDS}fold_splits.json"
    
    if os.path.exists(folds_path):
        print(f"Loading existing patient-wise {N_FOLDS}-fold splits")
        with open(folds_path, "r") as f:
            folds = json.load(f)
    else:
        print(f"Creating new patient-wise {N_FOLDS}-fold splits")
        folds = get_patient_wise_folds(df, n_folds=N_FOLDS, seed=SEED)
        with open(folds_path, "w") as f:
            json.dump(folds, f, indent=4)
        print(f"Patient-wise fold splits saved to: {folds_path}")
    
    fold_results = []
    f1_scores = []
    bacc_scores = []
    
    for fold_idx in range(N_FOLDS):
        fold_data = folds[str(fold_idx)]
        result, f1, bacc = run_fold_experiment(fold_idx, fold_data, features, labels, input_dim, output_dim)
        fold_results.append(result)
        f1_scores.append(f1)
        bacc_scores.append(bacc)
    
    f1_mean = np.mean(f1_scores)
    f1_std = np.std(f1_scores)
    bacc_mean = np.mean(bacc_scores)
    bacc_std = np.std(bacc_scores)
    
    print("\n===== Individual Fold Results =====")
    for idx, result in enumerate(fold_results):
        print(f"Fold {idx+1}: F1 = {result['validation_f1']:.4f}, BACC = {result['validation_bacc']:.4f}")
    
    print("\n===== Summary Statistics =====")
    print(f"F1 Score: {f1_mean:.4f} ± {f1_std:.4f}")
    print(f"Balanced Accuracy: {bacc_mean:.4f} ± {bacc_std:.4f}")
    
    final_results = {
        "fold_results": fold_results,
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
        "n_folds": N_FOLDS,
        "patient_wise_folds_path": folds_path
    }
    
    results_path = f"{BASE_PATH}/luad_linear_probing_multi_seed_results.json"
    
    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=4)
    
    print(f"Results saved to {results_path}")

if __name__ == "__main__":
    main()