import argparse
import pickle
from tqdm import tqdm
import os
from LEP_AD_model import *
from preprocess_data import *
from train import *
from utils import *
import pandas as pd
import numpy as np
from rdkit import RDLogger
from rdkit import Chem
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import random
RDLogger.DisableLog('rdApp.*')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device : " , device)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
#---------- Define training arguments ---------- 

parser = argparse.ArgumentParser(
    description='Training script for the model. Supports configuration of dataset paths, model parameters, training hyperparameters, and enum_epochsuation settings.'
)
parser.add_argument('--data_name', type=str, default='davis', help='Dataset to use')
parser.add_argument(
    '--split',
    type=str,
    default=None,
    choices=['similar_split', 'dissimilar_split'],
    help='Specifies the data split strategy to use: choose from "similar_split", "dissimilar_split".'
)
parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
parser.add_argument('--seed', type=int, default=42, help='Seed')
parser.add_argument('--num_epochs', type=int, default=700, help='Number of epochs')
parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads used in Transformer-based graph layers.')
parser.add_argument('--pooling_func', type=str, default='no_pooling', help='pooling_func')
parser.add_argument('--n_heads_ct', type=int, default=8, help='Number of attention heads used in cross-attention layers.')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate to apply to layers')
parser.add_argument('--lr', type=float, default=0.0005, help='Learning Rate')
parser.add_argument('--hidden_dim', type=int, default=128, help='hidden dimensionin the model')
parser.add_argument('--hidden_dim_mlp', type=int, default=128, help='hidden dimensionin the model')
parser.add_argument('--esm_model', type=str, default='esm3',
                    help='Specifies the ESM protein language model to use. Options: "esm2", "esm3", "esmc300m", or "esmc600m".')
parser.add_argument('--GNN_layer', type=str, default='TransformerConv',
    help='Specifies the graph neural network architecture used to embed the drug structure. Options may include: "GCN", "GATv2", "TransformerConv", etc.'
)
parser.add_argument(
    '--fusion_strategy',
    type=str,
    default='attention',
    help='Specifies the fusion strategy used to combine protein and drug features. Common options include "concat", "sum", or "attention".'
)

args = parser.parse_args()
model_name = "LEP-AD"
data_name = args.data_name
split = args.split
data_path = f"/ibex/project/c2012/Reem/LEP-AD_Paper/data/{data_name}/"
batch_size = args.batch_size
hidden_dim = args.hidden_dim
esm_model = args.esm_model
n_heads = args.n_heads
GNN_layer = args.GNN_layer
dropout = args.dropout
fusion = args.fusion_strategy
lr = args.lr
num_epochs = args.num_epochs
hidden_dim_mlp = args.hidden_dim_mlp
n_heads_ct = args.n_heads_ct
pooling_func = args.pooling_func
seed_value = args.seed

#---------- Import dataset ----------

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
    dataset = dataset.loc[dataset.compound_iso_smiles.isin([smi for smi in smiles_list if len(smi) <= 600])]
    num_epochs = 400
    esm_file_path = os.path.join(data_path, "representations", "embed_with_esm3.pkl")
    with open(esm_file_path, "rb") as f:
        esm3_embed = pickle.load(f)
    dataset = dataset.loc[dataset.target_sequence.isin(list(esm3_embed.keys()))]
    
print(
    f"\n📂 Dataset: {data_name} — Total samples: {len(dataset)}"
    f"\n🧬 ESM model: {esm_model}"
    f"\n🔗 Fusion strategy: {fusion}"
    f"\n🧮 Batch size: {batch_size}"
    f"\n💡 Hidden dim: {hidden_dim}"
    f"\n🧠 Hidden dim MLP: {hidden_dim_mlp}"
    f"\n🎯 n_heads_ct: {n_heads_ct}"
    f"\n🎯 n_heads: {n_heads}"
    f"\n🔍 GNN layer: {GNN_layer}"
    f"\n📉 Learning rate: {lr}"
    f"\n💧 Dropout: {dropout}"
    f"\n🧩 Pooling function: {pooling_func}",
    f"\n🧩 Seed: {seed_value}",
    flush=True
)

#---------- Create SMILE graphs ---------- 

drugs = []
smiles_list = dataset.compound_iso_smiles.unique().tolist()
for d in smiles_list:
    lg = Chem.MolToSmiles(Chem.MolFromSmiles(d), isomericSmiles=True)
    drugs.append(lg)
    
compound_iso_smiles = drugs
graph_path = os.path.join(data_path, 'representations', 'smile_graph.pickle')

if not os.path.exists(graph_path):
    print("Creating SMILES graphs...")
    os.makedirs(os.path.join(data_path, data_name, 'representations'), exist_ok=True)
    smile_graph = {}
    for smile in tqdm(compound_iso_smiles, desc="Processing SMILES"):
        g = smile_to_graph(smile)
        smile_graph[smile] = g
    with open(graph_path, 'wb') as handle:
        pickle.dump(smile_graph, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("SMILES graphs saved to:", graph_path)
else:
    print("Loading existing SMILES graphs from:", graph_path)

with open(graph_path, 'rb') as f:
    smile_graph = pickle.load(f)

    
#---------- Get protein representation ---------- 
esm_dim = 0
if esm_model == "esm2":
    esm_dim = 2560
    file_path = os.path.join(data_path, "representations", "protein_rep.pickle")
    with open(file_path, "rb") as f:
        target_reps_dict = pickle.load(f)
elif esm_model == "esmc300m":
    esm_dim = 960
    file_path = os.path.join(data_path, "representations", "embed_with_esmc_300m.pkl")
    with open(file_path, "rb") as f:
        target_reps_dict = pickle.load(f)
elif esm_model == "esmc600m":
    esm_dim = 1152
    file_path = os.path.join(data_path, "representations", "embed_with_esmc_600m.pkl")
    with open(file_path, "rb") as f:
        target_reps_dict = pickle.load(f)
elif esm_model == "esm3":
    esm_dim = 1536
    file_path = os.path.join(data_path, "representations", "embed_with_esm3.pkl")
    with open(file_path, "rb") as f:
        target_reps_dict = pickle.load(f)
else:
    raise ValueError(f"Unsupported esm_model: {esm_model}")

target_reps_dict = {k: v.mean(dim = 0) for k, v in target_reps_dict.items()}

#---------- Create DataLoaders ---------- 

# Determine which data splits to use
splits = [split] if split is not None else [
    'similar_split',
    'dissimilar_split',
]

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

    train_seqs = df_train.target_sequence.unique().tolist() 
    test_seqs = list(set(df_test.target_sequence.tolist() + df_valid.target_sequence.tolist()))
    target_reps_dict_norm = normalize_esm_embeddings_dim1(target_reps_dict, train_seqs , test_seqs)
    
    train_drugs = np.asarray(df_train['compound_iso_smiles'])
    train_prot_keys = np.asarray(df_train['target_sequence'])
    train_Y = np.asarray(df_train['affinity'])

    val_drugs = np.asarray(df_valid['compound_iso_smiles'])
    val_prot_keys = np.asarray(df_valid['target_sequence'])
    val_Y = np.asarray(df_valid['affinity'])

    test_drugs = np.asarray(df_test['compound_iso_smiles'])
    test_prot_keys = np.asarray(df_test['target_sequence'])
    test_Y = np.asarray(df_test['affinity'])

    set_seed(seed_value)
    
    train_dataset = DTADataset(
        root=data_path,
        dataset=data_name,
        xd=train_drugs,
        target_key=train_prot_keys,
        y=train_Y,
        smile_graph=smile_graph,
        target_rep=target_reps_dict_norm,
    )

    val_dataset = DTADataset(
        root=data_path,
        dataset=data_name,
        xd=val_drugs,
        target_key=val_prot_keys,
        y=val_Y,
        smile_graph=smile_graph,
        target_rep=target_reps_dict_norm,
    )

    test_dataset = DTADataset(
        root=data_path,
        dataset=data_name,
        xd=test_drugs,
        target_key=test_prot_keys,
        y=test_Y,
        smile_graph=smile_graph,
        target_rep=target_reps_dict_norm,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate)
    valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate)
    
    
    model = GNNNet(
            n_output=1,
            num_features_mol=78,
            hidden_dim=hidden_dim,
            hidden_dim_mlp = hidden_dim_mlp,
            n_heads_ct = n_heads_ct,
            dropout=dropout,
            esm_dim=esm_dim,
            n_heads=n_heads,
            GNN_layer=GNN_layer,
            fusion=fusion,
            pooling_func = pooling_func,
            seed = seed_value
    ).to(device)
    
    # total parameters
    total_params = sum(p.numel() for p in model.parameters())
    # trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}  |  Trainable: {trainable_params:,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_mse = float('inf')
    patience = 100
    no_improve_epochs = 0
    model_save_path = f"/ibex/project/c2012/Reem/LEP-AD_Paper/models/model_checkpoint_ct_HPO_{data_name}_{split}_{model_name}_{esm_model}_{fusion}_seed_{seed_value}.pt"
    torch.cuda.empty_cache()

    for epoch in tqdm(range(num_epochs), desc=f"Training {split}"):
        train(model, device, train_loader, optimizer, epoch + 1, batch_size)

        G_val, P_val = predicting(model, device, valid_loader)
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
    G, P = predicting(model, device, test_loader)
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
    output_path = f"/ibex/project/c2012/Reem/LEP-AD_Paper/results/results_ct_HPO_{data_name}_{split}_{model_name}_{esm_model}_{fusion}_seed_{seed_value}.csv"
    results_df.to_csv(output_path, index=False)
    print(f"✅ Test results saved to: {output_path}")