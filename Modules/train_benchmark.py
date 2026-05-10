import sys
from pathlib import Path

# Add the directory containing this script to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import os
import pickle
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch_geometric.loader import DataLoader
from rdkit import RDLogger
from rdkit import Chem
from model import *
from preprocess_data import *
import random
from train import *
from utils import *

# Featurization modules
from featurize.CSDTI_feat import *
from featurize.GraphDTA_feat import *
from featurize.IMAEN_feat import *
from featurize.MolTrans_feat import *
from featurize.GTB_DTI_feat import *

from featurize.CSDTI import *
from featurize.IMAEN import *
from featurize.TDGraphDTA import *
from featurize.FOTFCPI import *
from featurize.GTB_DTI import *
# Disable RDKit warnings
RDLogger.DisableLog('rdApp.*')


def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# Argument parser
parser = argparse.ArgumentParser(
    description='Training script for the model. Supports configuration of dataset paths, model parameters, training hyperparameters, and evaluation settings.'
)
parser.add_argument('--project_folder', type=str, required=True, help='Path to the project folder')
parser.add_argument('--data_name', type=str, default='davis', help='Dataset to use')
parser.add_argument('--Exper_evaluation', action='store_true', default=False)
parser.add_argument('--model_name', type=str, default='CSDTI', help='Model to run')
parser.add_argument(
    '--split',
    type=str,
    default=None,
    choices = ['similar_split', 'dissimilar_split'],
    help='Specifies the data split strategy to use: choose from "similar_split", or "dissimilar_split".'
)
parser.add_argument('--seed_value', type=int, default=42)

args = parser.parse_args()
project_folder = args.project_folder
Exper_evaluation = args.Exper_evaluation
data_name = args.data_name
model_name = args.model_name
split = args.split
seed_value = args.seed_value


print("Dataset:", data_name, "| Model:", model_name, "| Seed:", seed_value)

n_output = 1
data_path = f"{project_folder}/data/{data_name}/"
dataset_train = pd.read_csv(data_path + f"{data_name}_train.csv")
dataset_test = pd.read_csv(data_path + f"{data_name}_test.csv")
dataset = pd.concat([dataset_train, dataset_test], ignore_index=True)

smiles_list = dataset["compound_iso_smiles"].unique().tolist()


with open(data_path + "testing_proteins.pkl", "rb") as f:
    testing_proteins = pickle.load(f)
with open(data_path + "testing_drugs.pkl", "rb") as f:
    testing_drugs = pickle.load(f)

# Build sets (faster .isin)
pro_dis = set(testing_proteins.get("protein_dissimilar_split", []))
pro_sim = set(testing_proteins.get("protein_similar_split", []))
drug_dis = set(testing_drugs.get("drug_dissimilar_split", []))
drug_sim = set(testing_drugs.get("drug_similar_split", []))

# ---- DISSIMILAR SPLIT ----
dataset["dissimilar_split"] = "Train"

# Step 1: mark test proteins
mask_dis_pro = dataset["target_sequence"].isin(pro_dis)
dataset.loc[mask_dis_pro, "dissimilar_split"] = "Test"

# Step 2: mark test drugs
mask_dis_drug = dataset["compound_iso_smiles"].isin(drug_dis)
dataset.loc[mask_dis_drug, "dissimilar_split"] = "Test"


# ---- SIMILAR SPLIT ----
dataset["similar_split"] = "Train"

# Step 1: mark test proteins
mask_sim_pro = dataset["target_sequence"].isin(pro_sim)
dataset.loc[mask_sim_pro, "similar_split"] = "Test"

# Step 2: mark test drugs
mask_sim_drug = dataset["compound_iso_smiles"].isin(drug_sim)
dataset.loc[mask_sim_drug, "similar_split"] = "Test"

print("similar_split", dataset.similar_split.value_counts())
print("dissimilar_split", dataset.dissimilar_split.value_counts())

# For similar split
assert pro_sim.issubset(set(dataset.loc[dataset.similar_split == "Test"].target_sequence))
assert drug_sim.issubset(set(dataset.loc[dataset.similar_split == "Test"].compound_iso_smiles))

# For dissimilar split
assert pro_dis.issubset(set(dataset.loc[dataset.dissimilar_split == "Test"].target_sequence))
assert drug_dis.issubset(set(dataset.loc[dataset.dissimilar_split == "Test"].compound_iso_smiles))

if data_name == "Stitch":
    # Load ESM representations
    file_path = os.path.join(data_path, "representations", "embed_with_esm3.pkl")
    with open(file_path, "rb") as f:
        target_reps_dict = pickle.load(f)
    target_reps_dict = {k: (v).numpy().mean(0) for k, v in target_reps_dict.items()}
    dataset = dataset.loc[dataset.compound_iso_smiles.isin([smi for smi in smiles_list if len(smi) <= 600])]
    dataset = dataset.loc[dataset.target_sequence.isin(list(target_reps_dict.keys()))]

if Exper_evaluation:
    base_path = Path(f"{project_folder}/data")
    data_names = ["kiba", "ToxCast", "DTC", "Metz", "davis"]

    # Load training datasets
    dataset = pd.concat(
        [
            pd.concat(
                [
                    pd.read_csv(base_path / name / f"{name}_train.csv"),
                    pd.read_csv(base_path / name / f"{name}_test.csv"),
                ],
                ignore_index=True,
            )
            for name in data_names
        ],
        ignore_index=True,
    )
    dataset["Exper_evaluation_split"] = "Train"

    # Load experimental validation dataset
    testing_dataset = pd.read_csv(base_path / "expr_data.csv")
    testing_dataset["Exper_evaluation_split"] = "Test"

    # Merge train + experimental test
    dataset = pd.concat([dataset, testing_dataset], ignore_index=True)
    # dataset = dataset.loc[dataset.compound_iso_smiles.isin([smi for smi in smiles_list if len(smi) <= 600])]
    # dataset = dataset.loc[dataset.target_sequence.isin(list(target_reps_dict.keys()))]
    # num_epochs = 400 
    
print(f"\n📂 Dataset: {data_name} — Total samples: {len(dataset)}", flush=True)

# Load YAML config
config_path = f"{project_folder}/featurize/{model_name}.yaml"
with open(config_path, 'r') as f:
    cfg = yaml.safe_load(f)

# Required config fields
try:
    model_params = cfg['task']['model']['param']
    lr = float(cfg['optimizer']['lr'])
    batch_size = cfg['engine']['batch_size']
    if data_name == "Stitch":
        num_epochs = 400 #cfg['train']['num_epoch']
    else:
        num_epochs = 700
except KeyError as e:
    raise ValueError(f"Missing required config key: {e}")

# Featurization function mapping
featurize_map = {
    "CSDTI": CSDTI_featurize,
    "IMAEN": GraphDTA_featurize,
    "FOTFCPI": MolTrans_featurize,
    "TDGraphDTA": GraphDTA_featurize,
    "GTB_DTI": GTB_DTI_featurize
}

# Get featurizer
if model_name in featurize_map:
    featurize = featurize_map[model_name]()
else:
    raise ValueError(f"Unknown model name: {model_name}")

# Determine which data splits to use
if Exper_evaluation == False:
    splits = [split] if split is not None else [
        'similar_split',
        'dissimilar_split',
    ]
else: 
    splits = ["Exper_evaluation_split"]


results = []

for split in splits:
    print(f"\n=== Processing data split: '{split}' — Training for {num_epochs} epochs ===")

    # Full training and testing data for the current split
    df_train_full = dataset.loc[dataset[split] == 'Train'].copy()
    df_test = dataset.loc[dataset[split] == 'Test'].copy()
    
    df_train, df_valid = train_test_split(
        df_train_full,
        test_size=0.10,
        random_state=42,
        shuffle=True
    )
    
    # ✅ Move feature extraction here
    train_drugs = df_train['compound_iso_smiles'].tolist()
    train_prots = df_train['target_sequence'].tolist()
    train_Y = df_train['affinity'].tolist()

    val_drugs = df_valid['compound_iso_smiles'].tolist()
    val_prots = df_valid['target_sequence'].tolist()
    val_Y = df_valid['affinity'].tolist()

    test_drugs = df_test['compound_iso_smiles'].tolist()
    test_prots = df_test['target_sequence'].tolist()
    test_Y = df_test['affinity'].tolist()

    # Generate features
    train_set = featurize.data_input(train_drugs, train_prots, train_Y)
    val_set = featurize.data_input(val_drugs, val_prots, val_Y)
    test_set = featurize.data_input(test_drugs, test_prots, test_Y)

    print(f"Split: {split} | Train size: {len(train_set)} | Validation size: {len(val_set)} | Test size: {len(test_set)}")
    set_seed(seed_value)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    # Instantiate model
    if model_name == "CSDTI":
        set_seed(seed_value)
        model = CSDTI(**{**model_params, "n_output": n_output}).to(device)
    elif model_name == "IMAEN":
        set_seed(seed_value)
        model = IMAEN(**{**model_params, "n_output": n_output}).to(device)
    elif model_name == "FOTFCPI":
        set_seed(seed_value)
        model = FOTFCPI(**{**model_params, "n_output": n_output}).to(device)
    elif model_name == "TDGraphDTA":
        set_seed(seed_value)
        model = TDGraphDTA(**{**model_params, "n_output": n_output}).to(device)
    elif model_name == "GTB_DTI":
        set_seed(seed_value)
        model = GTB_DTI(**{**model_params, "n_output": n_output}).to(device)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val_mse = float('inf')
    patience = 100
    no_improve_epochs = 0
    model_save_path = f"{project_folder}/models/model_checkpoint_{data_name}_{split}_{model_name}_seed_{seed_value}.pt"
    torch.cuda.empty_cache()

    for epoch in tqdm(range(num_epochs), desc=f"Training {split}"):
        train_benchmark(model, device, train_loader, optimizer, epoch + 1, batch_size)

        G_val, P_val = predicting_benchmark(model, device, valid_loader)
        val_mse, _, _ = calculate_metrics(G_val, P_val, data_name)
        print(f"Epoch {epoch+1}: Validation MSE = {val_mse:.4f}")

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            no_improve_epochs = 0  # reset counter if improvement
            torch.save(model.state_dict(), model_save_path)
            print(f"✔️ New best model saved with val MSE = {best_val_mse:.4f}")
        else:
            no_improve_epochs += 1
            print(f"⚠️ No improvement for {no_improve_epochs} epoch(s)")
            
        if no_improve_epochs >= patience:
            print(f"⏹ Early stopping triggered after {patience} epochs without improvement")
            break

    # Final test
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    G, P = predicting_benchmark(model, device, test_loader)
    if Exper_evaluation:
        split = "Exper_evaluation_split"
        results_df = pd.DataFrame({
            "Split": split,
            "Model": model_name,
            "Dataset": data_name,
            "Compound": test_drugs,
            "True_affinity": test_Y,
            "Pred_affinity": P,
            "seed_value": seed_value
        })
        
        output_path = f"{project_folder}/results/results_{data_name}_{split}_{model_name}_{seed_value}.csv"
        results_df.to_csv(output_path, index=False)
        print(f"✅ Test results saved to: {output_path}")
    else:
        test_mse, test_ci, test_rm2 = calculate_metrics(G, P, data_name)
    
        print(f"{split} - MSE: {test_mse:.4f}, CI: {test_ci:.4f}, RM2: {test_rm2:.4f}")
    
        results.append({
            "Split": split,
            "Model": model_name,
            "Dataset": data_name,
            "Test_MSE": test_mse,
            "Test_CI": test_ci,
            "Test_RM2": test_rm2
        })
    
        # Save result for this split
        results_df = pd.DataFrame(results)
        output_path = f"{project_folder}/results/results_{data_name}_{split}_{model_name}_seed_{seed_value}.csv"
        results_df.to_csv(output_path, index=False)
        print(f"✅ Test results saved to: {output_path}")
