# All metrics
import os
import sys
import torch
import numpy as np
import pandas as pd
from random import shuffle
import matplotlib.pyplot as plt
from torch_geometric.data import Batch
from sklearn.preprocessing import StandardScaler
from scipy import stats
from lifelines.utils import concordance_index
import gzip
import pickle
from torch_geometric import data as DATA
import subprocess
from math import sqrt
from sklearn.metrics import average_precision_score
import time

#--------------------------------------------------------------------------------------------------------------

def to_matrix(seqs, emb_lookup):
    X, kept = [], []
    for s in seqs:
        v = emb_lookup[s]                     # e.g., target_reps_dict[s]
        # convert to numpy if torch tensor
        if hasattr(v, "detach"):
            v = v.detach().cpu().numpy()
        v = np.asarray(v, dtype=np.float32).ravel()
        X.append(v); kept.append(s)
    return np.stack(X, axis=0), kept

#--------------------------------------------------------------------------------------------------------------

def normalize_esm_embeddings(target_reps_dict, seqs, eps=1e-8):

    norm_dict = {}

    for s in seqs:
        v = target_reps_dict[s]

        # convert to numpy if it's a torch tensor
        if hasattr(v, "detach"):
            v = v.detach().cpu().numpy()

        v = np.asarray(v, dtype=np.float32)  # [L, 1536]

        # compute mean & std per feature (dimension 1536)
        mean = v.mean(axis=0, keepdims=True)      # [1, 1536]
        std  = v.std(axis=0, keepdims=True)       # [1, 1536]

        # avoid division by zero
        std = np.where(std < eps, eps, std)

        # normalize: same shape [L, 1536]
        v_norm = (v - mean) / std

        norm_dict[s] = v_norm.astype(np.float32)

    return norm_dict



#--------------------------------------------------------------------------------------------------------------

def to_matrix_dim1(seqs, emb_lookup):
    X, kept = [], []
    for s in seqs:
        v = emb_lookup[s]                     # e.g., target_reps_dict[s]
        # convert to numpy if torch tensor
        if hasattr(v, "detach"):
            v = v.detach().cpu().numpy()
        v = np.asarray(v, dtype=np.float32).ravel()
        X.append(v); kept.append(s)
    return np.stack(X, axis=0), kept

#--------------------------------------------------------------------------------------------------------------

def normalize_esm_embeddings_dim1(target_reps_dict, train_seqs, test_seqs):
    # 2) Build matrices
    X_train, train_seqs = to_matrix(train_seqs, target_reps_dict)
    if len(test_seqs) >= 1 :
        X_test,  test_seqs  = to_matrix(test_seqs,  target_reps_dict)
    
    # 3) Fit on train, transform both
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    if len(test_seqs) >= 1 :
        X_test_scaled  = scaler.transform(X_test)
    
    # 4) (Optional) Put back into DataFrames and combine
    cols = [f"f{i}" for i in range(X_train.shape[1])]
    df_train_scaled = pd.DataFrame(X_train_scaled, index=train_seqs, columns=cols).assign(split='Train')
    if len(test_seqs) >= 1 :
        df_test_scaled  = pd.DataFrame(X_test_scaled,  index=test_seqs,  columns=cols).assign(split='Test')
    
        # single combined dataframe (sequence as a column)
        df_both = (
            pd.concat([df_train_scaled, df_test_scaled])
              .reset_index()
              .rename(columns={'index': 'target_sequence'})
        )
    
    # 5) (Optional) Make a scaled dict if you prefer that format
    scaled_target_reps = {s: v for s, v in zip(train_seqs, X_train_scaled)}
    if len(test_seqs) >= 1 :
        scaled_target_reps.update({s: v for s, v in zip(test_seqs, X_test_scaled)})
    return scaled_target_reps

#--------------------------------------------------------------------------------------------------------------

def load_model(model_path):
    model = torch.load(model_path)
    return model

#--------------------------------------------------------------------------------------------------------------
    
def calculate_metrics(Y, P, dataset='davis'):
   
    cindex2 = concordance_index(Y, P)  
    rm2 = get_rm2(Y, P) 
    mse = get_mse(Y, P)

    print('metrics for ', dataset)
    print('cindex2', cindex2)
    print('rm2:', rm2)
    print('mse:', mse)
 
    return mse,cindex2,rm2

#--------------------------------------------------------------------------------------------------------------

def plot_density(Y, P, fold=0, dataset='davis'):
    plt.figure(figsize=(10, 5))
    plt.grid(linestyle='--')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.scatter(P, Y, color='blue', s=40)
    plt.title('density of ' + dataset, fontsize=30, fontweight='bold')
    plt.xlabel('predicted', fontsize=30, fontweight='bold')
    plt.ylabel('measured', fontsize=30, fontweight='bold')
    # plt.xlim(0, 21)
    # plt.ylim(0, 21)
    if dataset == 'davis':
        plt.plot([5, 11], [5, 11], color='black')
    else:
        plt.plot([6, 16], [6, 16], color='black')
    plt.legend(loc=0, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=12, fontweight='bold')
    plt.savefig(os.path.join('results', dataset + '_' + str(fold) + '.png'), dpi=500, bbox_inches='tight')

#--------------------------------------------------------------------------------------------------------------

def get_aupr(Y, P, threshold=7.0):
    Y = np.where(Y >= 7.0, 1, 0)
    P = np.where(P >= 7.0, 1, 0)
    aupr = average_precision_score(Y, P)
    return aupr

#--------------------------------------------------------------------------------------------------------------

def get_cindex(Y, P):
    summ = 0
    pair = 0

    for i in range(1, len(Y)):
        for j in range(0, i):
            if i is not j:
                if (Y[i] > Y[j]):
                    pair += 1
                    summ += 1 * (P[i] > P[j]) + 0.5 * (P[i] == P[j])

    if pair != 0:
        return summ / pair
    else:
        return 0

#--------------------------------------------------------------------------------------------------------------

def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))

    return mult / float(y_obs_sq * y_pred_sq)

#--------------------------------------------------------------------------------------------------------------

def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs * y_pred) / float(sum(y_pred * y_pred))

#--------------------------------------------------------------------------------------------------------------

def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k * y_pred)) * (y_obs - (k * y_pred)))
    down = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))

    return 1 - (upp / float(down))

#--------------------------------------------------------------------------------------------------------------

def get_rm2(ys_orig, ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)

    return r2 * (1 - np.sqrt(np.absolute((r2 * r2) - (r02 * r02))))

#--------------------------------------------------------------------------------------------------------------

def get_rmse(y, f):
    rmse = sqrt(((y - f) ** 2).mean(axis=0))
    return rmse

#--------------------------------------------------------------------------------------------------------------

# def get_mse(y, f):
#     mse = ((y - f) ** 2).mean(axis=0)
#     return mse
def get_mse(y, f, eps=1e-12):
    mse = ((y - f) ** 2).mean(axis=0)
    var = y.var(axis=0, ddof=0)
    return mse / (var + eps)

#--------------------------------------------------------------------------------------------------------------

def get_pearson(y, f):
    rp = np.corrcoef(y, f)[0, 1]
    return rp

#--------------------------------------------------------------------------------------------------------------


def get_spearman(y, f):
    rs = stats.spearmanr(y, f)[0]
    return rs

#--------------------------------------------------------------------------------------------------------------

def get_ci(y, f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y) - 1
    j = i - 1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z + 1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i - 1
    ci = S / z
    return ci
