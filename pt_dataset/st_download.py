import os
import os
import pandas as pd
import datasets
from tqdm import tqdm
from pathlib import Path
from huggingface_hub import login

BASE_DIR = "./pt_dataset"
HEST_DATA_DIR = os.path.join(BASE_DIR, "hest_data")

login(token="Insert your token")

meta_df = pd.read_csv("hf://datasets/MahmoodLab/hest/HEST_v1_1_0.csv")
meta_df = meta_df[meta_df["species"] == "Homo sapiens"]
ids_to_query = meta_df["id"].values

list_h5_patterns = [f"*{id}.h5" for id in ids_to_query]

h5_dataset = datasets.load_dataset(
    'MahmoodLab/hest', 
    cache_dir=HEST_DATA_DIR,
    patterns=list_h5_patterns
)