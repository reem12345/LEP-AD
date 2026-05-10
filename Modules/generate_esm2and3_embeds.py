from huggingface_hub import login
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence, Dict, List, Tuple

import os
import gc
import pickle
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

import esm
from esm.pretrained import LOCAL_MODEL_REGISTRY, load_local_model
from esm.models.esmc import ESMC
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, LogitsConfig
from transformers import AutoModel, AutoTokenizer

# ---------------------------------- Add HuggingFace token ----------------------------------
#login(token="")

print("Available ESM models:")
for model_name in LOCAL_MODEL_REGISTRY.keys():
    print("-", model_name)

import argparse

@torch.no_grad()
def embed_with_esmc_300m(
    seq_list: List[str],
    data_name: str,
    save_root: str,
    device: str = "cuda",
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"

    model = ESMC.from_pretrained("esmc_300m").to(device).eval()

    embeddings_dict: Dict[str, np.ndarray] = {}
    failed_seqs: List[str] = []

    for i, seq in enumerate(tqdm(seq_list, desc="Embedding with ESMC-300M")):
        protein_tensor = None
        logits_output = None
        emb = None
        try:
            protein = ESMProtein(sequence=seq)
            protein_tensor = model.encode(protein)
            if hasattr(protein_tensor, "to"):
                protein_tensor = protein_tensor.to(device)

            logits_output = model.logits(
                protein_tensor,
                LogitsConfig(sequence=True, return_embeddings=True)
            )
            # (B=1, L, D) → (L, D)
            emb = logits_output.embeddings.squeeze(0)
            embeddings_dict[seq] = emb.detach().cpu().float()

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"❌ CUDA OOM at index {i}; skipping")
                failed_seqs.append(seq)
                try:
                    del protein_tensor, logits_output, emb
                except Exception:
                    pass
                torch.cuda.empty_cache(); gc.collect()
                continue
            else:
                raise
        except Exception as e:
            print(f"⚠️ Failed at index {i}: {e}")
            failed_seqs.append(seq)
            continue

        if use_cuda and (i % 100 == 0):
            try:
                del protein_tensor, logits_output, emb
            except Exception:
                pass
            torch.cuda.empty_cache(); gc.collect()

    save_dir = os.path.join(save_root, data_name, "representations")
    os.makedirs(save_dir, exist_ok=True)
    embed_path = os.path.join(save_dir, "embed_with_esmc_300m.pkl")
    with open(embed_path, "wb") as f:
        pickle.dump(embeddings_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✅ Saved {len(embeddings_dict)} embeddings to: {embed_path}")
    if failed_seqs:
        print(f"⚠️ Failed sequences: {len(failed_seqs)}")
    return embeddings_dict, failed_seqs


#------------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def embed_with_esmc_600m(
    seq_list: List[str],
    data_name: str,
    save_root: str,
    device: str = "cuda",
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"

    model = ESMC.from_pretrained("esmc_600m").to(device).eval()

    embeddings_dict: Dict[str, np.ndarray] = {}
    failed_seqs: List[str] = []

    for i, seq in enumerate(tqdm(seq_list, desc="Embedding with ESMC-600M")):
        protein_tensor = None
        logits_output = None
        emb = None
        try:
            protein = ESMProtein(sequence=seq)
            protein_tensor = model.encode(protein)
            if hasattr(protein_tensor, "to"):
                protein_tensor = protein_tensor.to(device)

            logits_output = model.logits(
                protein_tensor,
                LogitsConfig(sequence=True, return_embeddings=True)
            )
            # (B=1, L, D) → (L, D)
            emb = logits_output.embeddings.squeeze(0)
            embeddings_dict[seq] = emb.detach().cpu().float()

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"❌ CUDA OOM at index {i}; skipping")
                failed_seqs.append(seq)
                try:
                    del protein_tensor, logits_output, emb
                except Exception:
                    pass
                torch.cuda.empty_cache(); gc.collect()
                continue
            else:
                raise
        except Exception as e:
            print(f"⚠️ Failed at index {i}: {e}")
            failed_seqs.append(seq)
            continue
        finally:
            if use_cuda and (i % 100 == 0):
                try:
                    del protein_tensor, logits_output, emb
                except Exception:
                    pass
                torch.cuda.empty_cache(); gc.collect()

    save_dir = os.path.join(save_root, data_name, "representations")
    os.makedirs(save_dir, exist_ok=True)
    embed_path = os.path.join(save_dir, "embed_with_esmc_600m.pkl")
    with open(embed_path, "wb") as f:
        pickle.dump(embeddings_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✅ Saved {len(embeddings_dict)} embeddings to: {embed_path}")
    if failed_seqs:
        print(f"⚠️ Failed sequences: {len(failed_seqs)}")
    return embeddings_dict, failed_seqs


#------------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def embed_with_esm3(
    seq_list: List[str],
    data_name: str,
    save_root: str,
    model_name: str = "esm3-sm-open-v1",
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    save_every: int = 100
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"

    save_dir = os.path.join(save_root, data_name, "representations")
    os.makedirs(save_dir, exist_ok=True)
    embed_path  = os.path.join(save_dir, "embed_with_esm3.pkl")
    failed_path = os.path.join(save_dir, "embed_with_esm3_failed.pkl")

    if os.path.exists(embed_path):
        with open(embed_path, "rb") as f:
            embeddings_dict: Dict[str, np.ndarray] = pickle.load(f)
        print(f"📂 Loaded {len(embeddings_dict)} existing embeddings")
    else:
        embeddings_dict = {}
    if os.path.exists(failed_path):
        with open(failed_path, "rb") as f:
            failed_seqs: List[str] = pickle.load(f)
    else:
        failed_seqs = []

    model = ESM3.from_pretrained(model_name).to(device).to(dtype=dtype).eval()

    for i, seq in enumerate(tqdm(seq_list, desc=f"Embedding with {model_name}")):
        if seq in embeddings_dict:
            continue
        protein_tensor = logits_output = emb = None
        try:
            protein = ESMProtein(sequence=seq)
            protein_tensor = model.encode(protein)
            if hasattr(protein_tensor, "to"):
                protein_tensor = protein_tensor.to(device)
            logits_output = model.logits(
                protein_tensor,
                LogitsConfig(sequence=True, return_embeddings=True)
            )
            # (B=1, L, D) → (L, D)
            emb = logits_output.embeddings.squeeze(0)
            embeddings_dict[seq] = emb.detach().cpu().float()
        except Exception as e:
            print(f"⚠️ Failed at index {i}: {e}")
            failed_seqs.append(seq)

        if (i + 1) % save_every == 0 or i == len(seq_list) - 1:
            with open(embed_path, "wb") as f:
                pickle.dump(embeddings_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(failed_path, "wb") as f:
                pickle.dump(failed_seqs, f, protocol=pickle.HIGHEST_PROTOCOL)
            if use_cuda:
                try:
                    del protein_tensor, logits_output, emb
                except Exception:
                    pass
                torch.cuda.empty_cache()
            gc.collect()

    print(f"✅ Saved {len(embeddings_dict)} embeddings to: {embed_path}")
    print(f"⚠️ Failed sequences so far: {len(failed_seqs)} → {failed_path}")
    return embeddings_dict, failed_seqs


#------------------------------------------------------------------------------------------------------------------------

@torch.no_grad()
def embed_with_esm2(
    seq_list: List[str],
    data_name: str,
    save_root: str,
    device: str = "cuda",
    model_name: str = "facebook/esm2_t36_3B_UR50D",  # Any HF ESM-2 model
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Generate full per-residue embeddings (L, D) using Hugging Face ESM-2 models.
    Works for any ESM-2 model on HF (e.g., facebook/esm2_t33_650M_UR50D, facebook/esm2_t36_3B_UR50D).
    Saves embeddings as NumPy float32 arrays to pickle.
    """
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"

    print(f"📥 Loading Hugging Face model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    embeddings_dict: Dict[str, np.ndarray] = {}
    failed_seqs: List[str] = []

    print(f"🚀 Generating embeddings for {len(seq_list):,} sequences...")
    for i, seq in enumerate(tqdm(seq_list, desc=f"Embedding with {model_name}")):
        seq_clean = (seq or "").strip().upper().replace(" ", "")
        try:
            # Tokenize
            enc = tokenizer(seq_clean, return_tensors="pt", add_special_tokens=True)
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            # Forward pass
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            reps = out.last_hidden_state  # shape [1, T, D] incl. special tokens

            # Remove BOS/EOS → (L, D)
            L = len(seq_clean)
            per_res = reps[:, 1:L+1, :].squeeze(0)  # keep only amino acid positions
            embeddings_dict[seq] = per_res.detach().cpu().float()

        except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f"⚠️ CUDA OOM at index {i}, skipping sequence.")
            failed_seqs.append(seq)
            torch.cuda.empty_cache()
        else:
            raise
        except Exception as e:
            print(f"⚠️ Failed at index {i}: {e}")
            failed_seqs.append(seq)
        finally:
            if use_cuda and (i % 100 == 0):
                torch.cuda.empty_cache(); gc.collect()

    # Save to file
    save_dir = os.path.join(save_root, data_name, "representations")
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"embed_with_esm2.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(embeddings_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"✅ Saved {len(embeddings_dict)} embeddings to: {out_path}")
    if failed_seqs:
        print(f"⚠️ Failed {len(failed_seqs)} sequences (see list in memory).")

    return embeddings_dict, failed_seqs


#------------------------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description='Generate protein representations using esm2, esmc, and esm3'
)
parser.add_argument('--project_folder', type=str, required=True, help='Path to the project folder')
parser.add_argument('--data_name', type=str, default='davis', help='Dataset to use')
args = parser.parse_args()
data_name = args.data_name
project_folder = args.project_folder
# Load the file — assuming it's whitespace- or tab-separated
df_train = pd.read_csv(f"{project_folder}/data/{data_name}/{data_name}_train.csv")
df_test = pd.read_csv(f"{project_folder}/data/{data_name}/{data_name}_test.csv")

# Concatenate along the rows (i.e., stack |test data below train data)
df = pd.concat([df_train, df_test], ignore_index=True)
print(data_name , df.columns)
print( len(sorted(df["target_sequence"].unique().tolist())) )

embeddings_dict, failed = embed_with_esmc_300m(
    seq_list=sorted(df["target_sequence"].dropna().unique().tolist()),
    data_name=data_name,
    save_root=f"{project_folder}/data"
)

embeddings_dict, failed = embed_with_esmc_600m(
    seq_list=sorted(df["target_sequence"].dropna().unique().tolist()),
    data_name=data_name,
    save_root=f"{project_folder}/data"
)

embeddings_dict, failed = embed_with_esm3(
    seq_list=sorted(df["target_sequence"].dropna().unique().tolist()),
    data_name=data_name,
    save_root=f"{project_folder}/data"
)

embeddings_dict, failed = embed_with_esm2(
    seq_list=sorted(df["target_sequence"].dropna().unique().tolist()),
    data_name=data_name,
    save_root=f"{project_folder}/data"
)

file_path = f"{project_folder}/data/{data_name}/representations/embed_with_esmc_300m.pkl"
with open(file_path, "rb") as f:
    embeddings_dict = pickle.load(f)
print(f"Loaded esmc300m embeddings for {len(embeddings_dict)} sequences")

file_path = f"{project_folder}/data/{data_name}/representations/embed_with_esmc_600m.pkl"
with open(file_path, "rb") as f:
    embeddings_dict = pickle.load(f)
print(f"Loaded esmc600m embeddings for {len(embeddings_dict)} sequences")

file_path = f"{project_folder}/data/{data_name}/representations/embed_with_esm3.pkl"
with open(file_path, "rb") as f:
    embeddings_dict = pickle.load(f)
print(f"Loaded esm3 embeddings for {len(embeddings_dict)} sequences")

file_path = f"{project_folder}/data/{data_name}/representations/embed_with_esm2.pkl"
with open(file_path, "rb") as f:
    embeddings_dict = pickle.load(f)
print(f"Loaded esm2 embeddings for {len(embeddings_dict)} sequences")
