import os
os.environ["OPENBLAS_NUM_THREADS"] = "32"
os.environ["OMP_NUM_THREADS"] = "32"
os.environ["MKL_NUM_THREADS"] = "32"
os.environ["VECLIB_MAXIMUM_THREADS"] = "32"
os.environ["NUMEXPR_NUM_THREADS"] = "32"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import random 
import re
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import umap
import warnings
warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")

sys.path.append("./utils")
from pt_load_inference import VisionInference

# Model configuration
VISION_MODEL_NAME = "pathoduet"  # Options: 'pathoduet', 'phikon', 'uni2_h'
LOSS_TYPE = "clip"
NUM_EPOCHS = 12
CHECKPOINT_PATH = f"./checkpoints/st_conma_pythia410m_12layers_3aug_clip/checkpoint_epoch_12.pt"

MODEL_NAME = "ST_ConMa"
BASE_DIR = "./results/cluster_plot/"
BASE_PATH = f"{BASE_DIR}st_conma"

os.makedirs(BASE_PATH, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ST-ConMa-text vision encoder once for all datasets
print("Loading ST-ConMa-text vision encoder...")
print(f"  Vision model: {VISION_MODEL_NAME}")
print(f"  Checkpoint: {CHECKPOINT_PATH}")

vision_model = VisionInference(
    checkpoint_path=CHECKPOINT_PATH,
    model_name=VISION_MODEL_NAME,
    device=device
)

def set_transform():
    """
    Use the transform from the vision model to ensure consistency

    Returns:
        img_transform: transform from the vision model
    """
    return vision_model.transform
    
def set_dataset(benchmark = None):
    if benchmark == "cam17":
        dataset = load_dataset("wltjr1007/Camelyon17-WILDS")
        train_data = dataset["train"]
        test_data = dataset["test"]

        return train_data, test_data

    if benchmark == "mhist":
        meta_data = pd.read_csv("./ft_dataset/linear_probing/mhist/annotations.csv")
        file_path = "./ft_dataset/linear_probing/mhist/images"
        image_paths = [os.path.join(file_path, img) for img in meta_data["Image Name"]]
        meta_data["Image Name"] = image_paths
        train_data = meta_data[meta_data["Partition"] == "train"]
        test_data = meta_data[meta_data["Partition"] == "test"]

        lb2id = {"HP" : 0, "SSA" : 1}

        return train_data, test_data, lb2id

    if benchmark == "crc":
        train_folder_path = "./ft_dataset/linear_probing/crc/NCT-CRC-HE-100K"
        test_folder_path = "./ft_dataset/linear_probing/crc/CRC-VAL-HE-7K"

        train_data = []
        test_data = []

        # Support multiple image extensions
        valid_extensions = (".tif", ".tiff", ".png", ".jpg", ".jpeg")

        for root, _, files in os.walk(train_folder_path):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    train_data.append(os.path.join(root, file))

        for root, _, files in os.walk(test_folder_path):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    test_data.append(os.path.join(root, file))

        print(f"CRC train files: {len(train_data)}, test files: {len(test_data)}")

        if len(test_data) == 0:
            raise ValueError(f"No image files found in {test_folder_path}. Check if the folder exists and contains valid image files.")

        return train_data, test_data

    if benchmark == "luad":
        data_folder_path = "./ft_dataset/linear_probing/luad/"

        file_list = []
        for root, _, files in os.walk(data_folder_path):
            for file in sorted(files):
                if file.endswith(".png"):
                    file_list.append(os.path.join(root, file))

        data = []
        for file in file_list:
            filename = os.path.basename(file)
            match = re.match(r"(.+)-[\d]+-[\d]+-\[(\d), (\d), (\d)\]\.png", filename) 
            if match:
                sample_id, label1, label2, label3 = match.groups()
                label = f"{label1},{label2},{label3}"
                if (label1, label2, label3).count('1') == 1:  
                    data.append((sample_id, file, label))
        
        df = pd.DataFrame(data, columns=["SampleID", "Filename", "Label"])
        df = df.sort_values("SampleID").reset_index(drop=True)

        df["LabelClass"] = df["Label"].apply(lambda x: x.split(",").index("1"))

        print(f"Total samples (patients): {len(df['SampleID'].unique())}")
        print(f"Total images: {len(df)}")
        print(f"Class distribution: {df['LabelClass'].value_counts().to_dict()}")
    
        return df
    
class cam17_Patch_Dataset(Dataset):
    def __init__(self, data, transforms=None):
        self.data = data
        self.img_transform = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]["image"]
        label = torch.tensor(self.data[idx]["label"], dtype=torch.float32)

        # Ensure image is RGB (convert from RGBA if needed)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.img_transform:
            transformed_img = self.img_transform(image)
        else:
            raise ValueError("Error: img_transform is None. Please provide a valid transformation.")

        return transformed_img, label

class crc_Patch_Dataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform
        self.label_map = self._get_labels()

    def _get_labels(self):
        labels = {}
        class_names = sorted(set([os.path.basename(os.path.dirname(path)) for path in self.file_paths]))
        class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
        for path in self.file_paths:
            label = os.path.basename(os.path.dirname(path))
            labels[path] = class_to_idx[label]
        return labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image_path = self.file_paths[idx]
        image = Image.open(image_path)

        # Ensure image is RGB (convert from RGBA if needed)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        label = self.label_map[image_path]
        if self.transform:
            transformed_img = self.transform(image)

        return transformed_img, torch.tensor(label, dtype=torch.long)

class luad_Patch_Dataset(Dataset):
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

class mhist_Patch_Dataset(Dataset):
    def __init__(self, df, lb2id, transforms=None):
        self.df = df
        self.lb2id = lb2id
        self.img_transform = transforms

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        image_path = self.df.iloc[idx]["Image Name"]
        image = Image.open(image_path).convert("RGB") 

        label = torch.tensor(self.lb2id[self.df.iloc[idx]["Majority Vote Label"]], dtype = torch.long)
        if self.img_transform:
            transformed_img = self.img_transform(image)
        else:
            raise ValueError("Error: img_transform is None. Please provide a valid transformation.")

        return transformed_img, label    

def extract_features(dataloader, benchmark = None):
    """
    Extract vision embeddings from ST-ConMa-text vision encoder

    Args:
        dataloader: DataLoader providing images
        benchmark: dataset name for label handling

    Returns:
        features: [N, vision_dim] numpy array
        labels: [N] numpy array
    """
    vision_model.model.eval()
    features, labels = [], []

    with torch.no_grad():
        for images, target in dataloader:
            images = images.to(device)

            # Extract vision embeddings
            # model returns (patch_features, representation)
            _, image_features = vision_model.model(images)  # [B, vision_dim]

            image_features = image_features.cpu().numpy()
            features.append(image_features)
            labels.append(target.numpy())

    features = np.vstack(features)

    if benchmark == "luad":
        labels = np.concatenate(labels, axis=0)
    else:
        labels = np.hstack(labels)

    return features, labels
    
def plot_umap_and_save(image_embeddings, true_labels, pred_labels, save_path_prefix, ari_score, benchmark):
    umap_model = umap.UMAP(n_components=2, random_state=42) 
    umap_embeddings = umap_model.fit_transform(image_embeddings)

    unique_true_labels = sorted(list(set(true_labels)))
    unique_pred_labels = sorted(list(set(pred_labels)))
    
    cmap = plt.get_cmap('tab10' if len(unique_true_labels) <= 10 else 'tab20' if len(unique_true_labels) <= 20 else 'gist_ncar')
    
    label_names = {}
    if benchmark == "cam17":
        label_names = {0: "Non-tumor", 1: "Tumor"}
    elif benchmark == "mhist":
        label_names = {0: "HP", 1: "SSA"}
    elif benchmark == "crc":
        label_names = {
            0: "ADI", 1: "BACK", 2: "DEB", 
            3: "LYM", 4: "MUC", 5: "MUS", 
            6: "NORM", 7: "STR", 8: "TUM"
        }
    elif benchmark == "luad":
        label_names = {0: "Tumor", 1: "Stroma", 2: "Normal"}

    plt.figure(figsize=(16, 7))
    plt.subplot(1, 2, 1)

    for i, label in enumerate(unique_true_labels):
        idx = true_labels == label 
        plt.scatter(
            umap_embeddings[idx, 0], 
            umap_embeddings[idx, 1], 
            c=[cmap(i / len(unique_true_labels))], 
            label=label_names.get(label, f"Class {label}"),
            alpha=0.6, 
            edgecolor='k', 
            linewidth=0.2 
        )
        
    plt.title("UMAP Projection of Image Embeddings (True Labels)") 
    plt.xlabel("UMAP Dim 1")
    plt.ylabel("UMAP Dim 2")
        
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') 
        
    plt.subplot(1, 2, 2)
        
    for i, label in enumerate(unique_pred_labels):
        idx = pred_labels == label
        plt.scatter(
            umap_embeddings[idx, 0], 
            umap_embeddings[idx, 1], 
            c=[cmap(i / len(unique_pred_labels))], 
            label=f"Cluster {label}",
            alpha=0.6, 
            edgecolor='k', 
            linewidth=0.2
        )
        
    plt.title(f"UMAP Projection (KMeans Clusters, ARI: {ari_score:.4f})")
    plt.xlabel("UMAP Dim 1")
    plt.ylabel("UMAP Dim 2")
        
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    os.makedirs(f"{save_path_prefix}/umap", exist_ok=True)
    plt.savefig(f"{save_path_prefix}/umap/{benchmark}_umap_clusters_ari{ari_score:.4f}.png", dpi=300, bbox_inches='tight')
    plt.close()

print("\n" + "="*70)
print("Evaluating CAM17 Dataset")
print("="*70)

cam17_img_transform = set_transform()
train_data, test_data = set_dataset(benchmark = "cam17")
test_dataset = cam17_Patch_Dataset(data=test_data, transforms = cam17_img_transform)

test_dataloader = DataLoader(test_dataset, batch_size=256, num_workers=4, shuffle=False, drop_last=False)

results = []

image_embeddings, true_labels = extract_features(test_dataloader, benchmark="cam17")
  
num_classes = len(set(true_labels))  
kmeans = KMeans(n_clusters=num_classes, random_state=42, n_init=50)
pred_labels = kmeans.fit_predict(image_embeddings)

ari_score = adjusted_rand_score(true_labels, pred_labels)
nmi_score = normalized_mutual_info_score(true_labels, pred_labels)
  
print(f"CAM17 - ARI: {ari_score:.4f} | NMI: {nmi_score:.4f}")

umap_model = umap.UMAP(n_components=2, random_state=42)
umap_embeddings = umap_model.fit_transform(image_embeddings)

plot_umap_and_save(
    image_embeddings=image_embeddings,
    true_labels=true_labels,
    pred_labels=pred_labels,
    save_path_prefix=BASE_PATH,
    ari_score=ari_score,
    benchmark="cam17"
)

results.append({
  "dataset" : "cam17",
  "ari": float(ari_score),
  "nmi": float(nmi_score)})

with open(f"{BASE_PATH}/cam17_clustering_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("CAM17 clustering evaluation results saved.")

print("\n" + "="*70)
print("Evaluating CRC Dataset")
print("="*70)

train_tif_files, test_tif_files = set_dataset(benchmark="crc")
crc_img_transform = set_transform()

test_dataset = crc_Patch_Dataset(test_tif_files, transform=crc_img_transform)

test_dataloader = DataLoader(test_dataset, batch_size=256, num_workers=4, shuffle=False, drop_last=False)

results = []

image_embeddings, true_labels = extract_features(test_dataloader, benchmark="crc")
  
num_classes = len(set(true_labels))  
kmeans = KMeans(n_clusters=num_classes, random_state=42, n_init=50)
pred_labels = kmeans.fit_predict(image_embeddings)
  
ari_score = adjusted_rand_score(true_labels, pred_labels)
nmi_score = normalized_mutual_info_score(true_labels, pred_labels)
  
print(f"CRC - ARI: {ari_score:.4f} | NMI: {nmi_score:.4f}")

plot_umap_and_save(
    image_embeddings=image_embeddings,
    true_labels=true_labels,
    pred_labels=pred_labels,
    save_path_prefix=BASE_PATH,
    ari_score=ari_score,
    benchmark="crc"
)

results.append({
  "dataset" : "crc",
  "ari": float(ari_score),
  "nmi": float(nmi_score)})

with open(f"{BASE_PATH}/crc_clustering_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("CRC clustering evaluation results saved.")

print("\n" + "="*70)
print("Evaluating MHIST Dataset")
print("="*70)

train_df, test_df, lb2id = set_dataset(benchmark="mhist")
mhist_img_transform = set_transform()

test_dataset = mhist_Patch_Dataset(df = test_df, lb2id = lb2id, transforms = mhist_img_transform)

test_dataloader = DataLoader(test_dataset, batch_size=256, num_workers=4, shuffle=False, drop_last=False)

results = []

image_embeddings, true_labels = extract_features(test_dataloader, benchmark="mhist")
  
num_classes = len(set(true_labels))  
kmeans = KMeans(n_clusters=num_classes, random_state=42, n_init=50)
pred_labels = kmeans.fit_predict(image_embeddings)
  
ari_score = adjusted_rand_score(true_labels, pred_labels)
nmi_score = normalized_mutual_info_score(true_labels, pred_labels)
  
print(f"MHIST - ARI: {ari_score:.4f} | NMI: {nmi_score:.4f}")

plot_umap_and_save(
    image_embeddings=image_embeddings,
    true_labels=true_labels,
    pred_labels=pred_labels,
    save_path_prefix=BASE_PATH,
    ari_score=ari_score,
    benchmark="mhist"
)

results.append({
  "dataset" : "mhist",
  "ari": float(ari_score),
  "nmi": float(nmi_score)})

with open(f"{BASE_PATH}/mhist_clustering_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("MHIST clustering evaluation results saved.")

print("\n" + "="*70)
print("Evaluating LUAD Dataset")
print("="*70)

luad_df = set_dataset(benchmark="luad")
luad_img_transform = set_transform()

luad_dataset = luad_Patch_Dataset(df=luad_df, transforms=luad_img_transform)

luad_dataloader = DataLoader(luad_dataset, batch_size=256, num_workers=4, shuffle=False, drop_last=False)

results = []

image_embeddings, true_labels = extract_features(luad_dataloader, benchmark="luad")

num_classes = len(set(true_labels))  
kmeans = KMeans(n_clusters=num_classes, random_state=42, n_init=50)
pred_labels = kmeans.fit_predict(image_embeddings)

ari_score = adjusted_rand_score(true_labels, pred_labels)
nmi_score = normalized_mutual_info_score(true_labels, pred_labels)

print(f"LUAD - ARI: {ari_score:.4f} | NMI: {nmi_score:.4f}")

plot_umap_and_save(
    image_embeddings=image_embeddings,
    true_labels=true_labels,
    pred_labels=pred_labels,
    save_path_prefix=BASE_PATH,
    ari_score=ari_score,
    benchmark="luad"
)

results.append({
    "dataset": "luad",
    "ari": float(ari_score),
    "nmi": float(nmi_score)
})

with open(f"{BASE_PATH}/luad_clustering_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("LUAD clustering evaluation results saved.")
