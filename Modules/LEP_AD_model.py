import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv,GATv2Conv , global_add_pool , global_mean_pool , global_max_pool
import sys
sys.path.insert(0, "/ibex/project/c2012/Reem/LEP-AD_Paper/")   
from torch_geometric.utils import to_dense_batch
import random, numpy as np
import os

# GCN based model
class GNNNet(torch.nn.Module):
    def __init__(self, n_output=1, num_features_mol=78, hidden_dim=128, 
                 hidden_dim_mlp = 128 , n_heads_ct = 2, 
                 dropout=0.2, esm_dim = 1152, 
                 n_heads = 8, GNN_layer = "TransformerConv", fusion = "concat", 
                 pooling_func = "global_add_pool", seed = 42, return_att_matrix = False):
        super(GNNNet, self).__init__()

        if fusion == "attention" and pooling_func != "no_pooling":
            raise ValueError("fusion='attention' requires pooling_func='no_pooling'")
        if fusion in ("concat", "sum") and pooling_func == "no_pooling":
            raise ValueError("fusion='concat'/'sum' requires a global pooling function, not 'no_pooling'")

        print('GNNNet Loaded')
        self.n_output = n_output
        self.seed = seed
        self.fusion = fusion
        self.pooling_func = pooling_func
        self.return_att_matrix = return_att_matrix
        sec_layer_n_heads = n_heads // 2
        th_layer_n_heads = n_heads // 4
        
        if GNN_layer == "TransformerConv":
            self.mol_conv1 = TransformerConv(
                in_channels = num_features_mol,
                out_channels = num_features_mol,
                heads = n_heads
            ) # num_features_mol * n_heads
        
            self.mol_conv2 = TransformerConv(
                in_channels = num_features_mol * n_heads, 
                out_channels = (num_features_mol * n_heads) // sec_layer_n_heads ,
                heads = sec_layer_n_heads
            )
            
            self.mol_conv3 = TransformerConv(
                in_channels = ( (num_features_mol * n_heads) // sec_layer_n_heads ) * sec_layer_n_heads,
                out_channels =   (num_features_mol * n_heads) // th_layer_n_heads  ,
                heads = th_layer_n_heads
            )
        
        elif GNN_layer == "GATv2":
            self.mol_conv1 = GATv2Conv(
                in_channels = num_features_mol,
                out_channels = num_features_mol,
                heads = n_heads,
                concat=True
            )
        
            self.mol_conv2 = GATv2Conv(
                in_channels = num_features_mol * n_heads,
                out_channels = (num_features_mol * n_heads) // sec_layer_n_heads ,
                heads = sec_layer_n_heads,
                concat=True
            )
        
            self.mol_conv3 = GATv2Conv(
                in_channels = ( (num_features_mol * n_heads) // sec_layer_n_heads ) * sec_layer_n_heads,
                out_channels =   (num_features_mol * n_heads) // th_layer_n_heads ,
                heads = th_layer_n_heads,
                concat=True
            )
        else:
            raise ValueError(f"Unsupported GNN model: {GNN_layer}")


            
        self.mol_fc_g1 = torch.nn.Linear(num_features_mol * n_heads, hidden_dim_mlp)
        self.mol_fc_g2 = torch.nn.Linear(hidden_dim_mlp, hidden_dim)
        self.pro_fc = nn.Linear(esm_dim, hidden_dim)

        if self.fusion == "attention":
            self.CrAtt_layer = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=n_heads_ct,
                batch_first=True,      # so we can keep [B, T, D]
            )
            
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        
        # combined layers
        if self.fusion == "concat" or self.fusion == "attention": 
            self.fc1 = nn.Linear(hidden_dim*2, hidden_dim_mlp)
        elif self.fusion == "sum":
            self.fc1 = nn.Linear(hidden_dim, hidden_dim_mlp)
            
        self.fc2 = nn.Linear(hidden_dim_mlp, hidden_dim_mlp // 2)
        self.out = nn.Linear(hidden_dim_mlp // 2, self.n_output)

    
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def forward(self, data_mol, data_pro):
        # get graph input
        mol_x, mol_edge_index, mol_batch = data_mol.x, data_mol.edge_index, data_mol.batch
        att_matrix1 = att_matrix2 = None
        
        x = self.mol_conv1(mol_x, mol_edge_index)
        x = self.relu(x)

        x = self.mol_conv2(x, mol_edge_index)
        x = self.relu(x)

        x = self.mol_conv3(x, mol_edge_index)
        x = self.relu(x)

        # global pooling # # (batch size , hidden_dim)
        drug_mask = None
        if self.pooling_func == "global_add_pool":
            x = global_add_pool(x, mol_batch)  
        elif self.pooling_func == "global_max_pool":
            x = global_max_pool(x, mol_batch)
        elif self.pooling_func == "global_mean_pool":
            x = global_mean_pool(x, mol_batch)
        elif self.pooling_func == "no_pooling":
            x, drug_mask = to_dense_batch(x, mol_batch)
            
        xc = x                                # (batch size , atoms, hidden_dim) or (batch size, hidden_dim)
        xc = self.relu(self.mol_fc_g1(xc))
        xc = self.dropout(xc)
        xc = self.mol_fc_g2(xc)
        xc = self.dropout(xc)
        x_pro = self.pro_fc(data_pro)        # (batch size, hidden_dim)
        
        if self.fusion == "concat":
            xc = torch.cat((xc , x_pro), 1)  # (batch size , 2 * hidden_dim)
            
        elif self.fusion == "sum":
            xc = xc + x_pro                  # (batch size , hidden_dim)
            
        elif self.fusion == "attention":
            # xc:    [B, T_d, D]   (drug atoms)
            # x_pro: [B, D]        (1D protein embedding)
            # drug_mask: [B, T_d]  (True = real, False = pad)
        
            # ---- Direction 1: protein queries drug ----
            # Make protein a sequence of length 1: [B, 1, D]
            prot_seq = x_pro.unsqueeze(1)        # [B, 1, D]
        
            # MHA wants True where positions are PAD
            key_padding_mask_drug = ~drug_mask   # [B, T_d]
        
            Dir1, att_matrix1 = self.CrAtt_layer(
                query=prot_seq,                  # [B, 1, D]
                key=xc,                          # [B, T_d, D]
                value=xc,                        # [B, T_d, D]
                key_padding_mask=key_padding_mask_drug,
                need_weights=True,
                average_attn_weights=False,      # [B, n_heads, 1, T_d]
            )
            # Dir1: [B, 1, D]
        
            # ---- Direction 2: drug queries protein ----
            # Protein side has length 1, no padding → no key_padding_mask needed
            Dir2, att_matrix2 = self.CrAtt_layer(
                query=xc,                        # [B, T_d, D]
                key=prot_seq,                    # [B, 1, D]
                value=prot_seq,                  # [B, 1, D]
                need_weights=True,
                average_attn_weights=False,      # [B, n_heads, T_d, 1]
            )
            # Dir2: [B, T_d, D]
        
            # ---- Pool each direction and fuse ----
            # Protein→drug: collapse length 1 → [B, D]
            Dir1_pool = Dir1.squeeze(1)          # [B, D]
        
            # Drug→protein: pool over atoms → [B, D]
            Dir2_pool = Dir2.amax(dim=1)         # [B, D]
            
            # Final fused representation
            xc = torch.cat((Dir1_pool, Dir2_pool), dim=1)   # [B, 2D]

            
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        
        if self.return_att_matrix:
            return out, att_matrix1, att_matrix2
        else:
            return out