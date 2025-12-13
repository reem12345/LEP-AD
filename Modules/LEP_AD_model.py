import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import TransformerConv, GATv2Conv, GCNConv, global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.utils import to_dense_batch
import random, numpy as np
import os

# GCN based model
class GNNNet(torch.nn.Module):
    def __init__(
        self,
        n_output=1,
        num_features_mol=78,
        hidden_dim=128,
        hidden_dim_mlp=128,
        n_heads_ct=2,
        dropout=0.2,
        esm_dim=1152,
        n_heads=8,
        GNN_layer="TransformerConv",
        fusion="concat",
        pooling_func="global_add_pool",
        seed=42,
        return_att_matrix=False
    ):
        """
        Drug–Protein regression model using a molecular GNN + protein embedding + fusion MLP.

        Parameters
        ----------
        n_output : int, default=1
            Output dimension (e.g., 1 for affinity regression).
        num_features_mol : int, default=78
            Number of atom/node features per molecule graph node (input feature size).
        hidden_dim : int, default=128
            Final latent dimension used for drug and protein representations before fusion.
        hidden_dim_mlp : int, default=128
            Hidden size used in intermediate MLP layers (drug projection and fusion head).
        n_heads_ct : int, default=2
            Number of heads for cross-attention (used only when fusion == "attention").
        dropout : float, default=0.2
            Dropout probability applied after projections and dense layers.
        esm_dim : int, default=1152
            Input dimension of the protein embedding (e.g., ESM embedding size).
        n_heads : int, default=8
            Number of attention heads in the molecular TransformerConv/GATv2 layers.
        GNN_layer : {"TransformerConv","GATv2"}, default="TransformerConv"
            Choice of molecular graph backbone layer type.
        fusion : {"concat","sum","attention"}, default="concat"
            How to fuse drug and protein representations:
            - "concat": concatenate [drug || protein]
            - "sum": elementwise sum (requires matching dims)
            - "attention": bidirectional cross-attention between drug atoms and protein embedding
        pooling_func : {"global_add_pool","global_max_pool","global_mean_pool","no_pooling"}, default="global_add_pool"
            Pooling strategy for the drug graph:
            - global_*: pools node embeddings to a graph embedding [B, F]
            - no_pooling: keeps atom-level sequence [B, T_d, F] using padding mask (needed for attention fusion)
        seed : int, default=42
            Random seed used to enforce determinism (sets Python/NumPy/PyTorch/CUDA seeds).
        return_att_matrix : bool, default=False
            If True, forward() returns (prediction, att_matrix1, att_matrix2).
            att_matrix1: protein→drug attention weights
            att_matrix2: drug→protein attention weights
        """
        super(GNNNet, self).__init__()

        print('GNNNet Loaded')
        self.n_output = n_output
        self.seed = seed
        self.fusion = fusion
        self.pooling_func = pooling_func
        self.return_att_matrix = return_att_matrix

        # Helper head counts used to control intermediate projection sizes
        # (used to choose out_channels so that final feature dim stays consistent)
        sec_layer_n_heads = n_heads // 2
        th_layer_n_heads = n_heads // 4

        # ---- Molecular GNN backbone (node-level) ----
        # Goal: produce node embeddings of size (num_features_mol * n_heads) at each layer output after concatenation.
        if GNN_layer == "TransformerConv":
            self.mol_conv1 = TransformerConv(
                in_channels=num_features_mol,
                out_channels=num_features_mol,
                heads=n_heads
            )  # output dim per node = num_features_mol * n_heads

            self.mol_conv2 = TransformerConv(
                in_channels=num_features_mol * n_heads,
                out_channels=(num_features_mol * n_heads) // sec_layer_n_heads,
                heads=sec_layer_n_heads
            )

            self.mol_conv3 = TransformerConv(
                in_channels=(((num_features_mol * n_heads) // sec_layer_n_heads) * sec_layer_n_heads),
                out_channels=(num_features_mol * n_heads) // th_layer_n_heads,
                heads=th_layer_n_heads
            )

        elif GNN_layer == "GATv2":
            self.mol_conv1 = GATv2Conv(
                in_channels=num_features_mol,
                out_channels=num_features_mol,
                heads=n_heads,
                concat=True
            )

            self.mol_conv2 = GATv2Conv(
                in_channels=num_features_mol * n_heads,
                out_channels=(num_features_mol * n_heads) // sec_layer_n_heads,
                heads=sec_layer_n_heads,
                concat=True
            )

            self.mol_conv3 = GATv2Conv(
                in_channels=(((num_features_mol * n_heads) // sec_layer_n_heads) * sec_layer_n_heads),
                out_channels=(num_features_mol * n_heads) // th_layer_n_heads,
                heads=th_layer_n_heads,
                concat=True
            )
        else:
            raise ValueError(f"Unsupported GNN model: {GNN_layer}")

        # ---- Projection heads ----
        # Drug projection:
        # - If pooled:   [B, num_features_mol*n_heads] -> [B, hidden_dim]
        # - If no_pooling: [B, T_d, num_features_mol*n_heads] -> [B, T_d, hidden_dim]
        self.mol_fc_g1 = torch.nn.Linear(num_features_mol * n_heads, hidden_dim_mlp)
        self.mol_fc_g2 = torch.nn.Linear(hidden_dim_mlp, hidden_dim)

        # Protein projection: [B, esm_dim] -> [B, hidden_dim]
        self.pro_fc = nn.Linear(esm_dim, hidden_dim)

        # ---- Cross-attention block (optional) ----
        # Only constructed if fusion == "attention"
        # Uses PyTorch MultiheadAttention in batch-first mode: [B, T, D]
        if self.fusion == "attention":
            self.CrAtt_layer = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=n_heads_ct,
                batch_first=True,  # keep [B, T, D] convention
            )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # ---- Fusion MLP head ----
        # concat/attention -> input is [B, 2*hidden_dim]
        # sum -> input is [B, hidden_dim]
        if self.fusion == "concat" or self.fusion == "attention":
            self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim_mlp)
        elif self.fusion == "sum":
            self.fc1 = nn.Linear(hidden_dim, hidden_dim_mlp)

        self.fc2 = nn.Linear(hidden_dim_mlp, hidden_dim_mlp // 2)
        self.out = nn.Linear(hidden_dim_mlp // 2, self.n_output)

        # ---- Determinism / reproducibility ----
        # Note: this sets seeds when the model is instantiated.
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def forward(self, data_mol, data_pro):
        """
        Forward pass.

        Parameters
        ----------
        data_mol : torch_geometric.data.Data (batched)
            Molecular graph batch containing:
            - data_mol.x           : [N_total_nodes, num_features_mol]
            - data_mol.edge_index  : [2, E_total_edges]
            - data_mol.batch       : [N_total_nodes] mapping node -> graph index
        data_pro : torch.Tensor
            Protein embedding tensor, typically:
            - [B, esm_dim]  (one embedding per protein)
            (Must match the expected esm_dim given to pro_fc.)

        Returns
        -------
        out : torch.Tensor
            Prediction tensor of shape [B, n_output].
        (optional) att_matrix1, att_matrix2 : torch.Tensor
            Returned only if self.return_att_matrix is True.
            - att_matrix1 : protein→drug attention weights, shape [B, n_heads_ct, 1, T_d]
            - att_matrix2 : drug→protein attention weights, shape [B, n_heads_ct, T_d, 1]
        """
        # get graph input
        mol_x, mol_edge_index, mol_batch = data_mol.x, data_mol.edge_index, data_mol.batch

        # Initialize attention matrices to None so return_att_matrix works for all fusion modes
        att_matrix1 = att_matrix2 = None

        # ---- Molecular message passing ----
        x = self.mol_conv1(mol_x, mol_edge_index)
        x = self.relu(x)

        x = self.mol_conv2(x, mol_edge_index)
        x = self.relu(x)

        x = self.mol_conv3(x, mol_edge_index)
        x = self.relu(x)

        # ---- Pooling / batching for drug representation ----
        # If global pooling: x -> [B, F]
        # If no_pooling:     x -> [B, T_d, F] and drug_mask -> [B, T_d]
        if self.pooling_func == "global_add_pool":
            x = global_add_pool(x, mol_batch)
        elif self.pooling_func == "global_max_pool":
            x = global_max_pool(x, mol_batch)
        elif self.pooling_func == "global_mean_pool":
            x = global_mean_pool(x, mol_batch)
        elif self.pooling_func == "no_pooling":
            x, drug_mask = to_dense_batch(x, mol_batch)

        # ---- Drug projection to hidden_dim ----
        # xc is either:
        # - [B, F] (pooled) OR
        # - [B, T_d, F] (no_pooling)
        xc = x
        xc = self.relu(self.mol_fc_g1(xc))
        xc = self.dropout(xc)
        xc = self.mol_fc_g2(xc)
        xc = self.dropout(xc)

        # ---- Protein projection to hidden_dim ----
        x_pro = self.pro_fc(data_pro)  # [B, hidden_dim]

        # ---- Fusion ----
        if self.fusion == "concat":
            # Concatenate graph-level drug vector and protein vector: [B, 2*hidden_dim]
            xc = torch.cat((xc, x_pro), 1)

        elif self.fusion == "sum":
            # Elementwise sum: [B, hidden_dim]
            xc = xc + x_pro

        elif self.fusion == "attention":
            # xc:    [B, T_d, D]   (drug atoms)
            # x_pro: [B, D]        (1D protein embedding)
            # drug_mask: [B, T_d]  (True = real, False = pad)

            # ---- Direction 1: protein queries drug ----
            # Make protein a sequence of length 1: [B, 1, D]
            prot_seq = x_pro.unsqueeze(1)  # [B, 1, D]

            # MHA wants True where positions are PAD
            key_padding_mask_drug = ~drug_mask  # [B, T_d]

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

            # Final fused representation: [B, 2D]
            xc = torch.cat((Dir1_pool, Dir2_pool), dim=1)

        # ---- Prediction head ----
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)

        # ---- Optional return of attention weights ----
        if self.return_att_matrix:
            return out, att_matrix1, att_matrix2
        else:
            return out

