

import numpy as np
import torch
import torch.nn as nn
from .le import LE
import sys
sys.path.append("..")
from utils.utils_graph import preprocess_nxgraph
from utils.walker import RandomWalker
from tqdm import tqdm
from collections import defaultdict
import networkx as nx

class MLP(nn.Module):
    """backbone"""
    def __init__(self, input_dim, hidden_dim, dropout=0.3):
        super(MLP, self).__init__()
        self.mlp_layers = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.GELU(), 
                    nn.BatchNorm1d(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(), 
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(dropout),
                    )

    def forward(self, x) :
        z = self.mlp_layers(x)
        return z


class Similarity(nn.Module):

    def __init__(self, input_dim, out_dim=1, activation_fn=None):
        super(Similarity, self).__init__()
        self.dense = nn.Linear(input_dim, out_dim)
        self.activation_fn = activation_fn

    def forward(self, x):
        # x is expected to be of shape [num_pairs, 2, feature_dim]
        l1_distance = torch.abs(x[:, 0, :] - x[:, 1, :]) 
        output = self.dense(l1_distance)
        if self.activation_fn:
            output = self.activation_fn(output) 
        return output

# ================== GraphMLP ==================
class GraphMLP(nn.Module):

    def __init__(self, x_pos_dim, x_dim=None, hidden_dim=128, dropout=0.3):
        super(GraphMLP, self).__init__()

        self.use_dual_encoding = x_dim is not None

        # 
        hidden_dim_pos = hidden_dim // 2 if self.use_dual_encoding else hidden_dim

        # 
        self.encoder_pos = MLP(x_pos_dim*3, hidden_dim_pos, dropout)

        # 
        self.encoder_feat = MLP(x_dim, hidden_dim // 2, dropout) if self.use_dual_encoding else None

        #
        self.similarity = Similarity(hidden_dim, activation_fn=torch.sigmoid)

    def reset_parameters(self):
        self.encoder_pos.reset_parameters()
        if self.encoder_feat is not None:
            self.encoder_feat.reset_parameters()



    # # 
    def embed(self, x_pos,x_feat=None):
        return self.dual_encoder(x_pos,x_feat)
    def dual_encoder(self, x_pos,x_feat=None):
        z_pos = self.encoder_pos(x_pos)
        z_feat = self.encoder_feat(x_feat) if (self.encoder_feat is not None and x_feat is not None) else None

        if z_feat is not None:
            z = torch.cat((z_pos,z_feat), dim=1)
        else:
            z = z_pos
        return z


    def similarity_from_embeddings(self, h_1, h_2):
        """
        h_1, h_2: [num_pairs, hidden_dim]
        """
        h = torch.stack((h_1, h_2), dim=1)  # [num_pairs, 2, hidden_dim]
        return self.similarity(h)

    def forward(self, x_pos_1, x_pos_2, x_feat_1=None, x_feat_2=None):
        h_1 = self.dual_encoder(x_pos_1, x_feat_1)
        h_2 = self.dual_encoder(x_pos_2, x_feat_2)
        h = torch.stack((h_1, h_2), dim=1)
        return self.similarity(h)

class DualViewGraphMLP(nn.Module):
    def __init__(self, graph_feat_dim, hyper_feat_dim, hidden_dim=128, dropout=0.3):
        super(DualViewGraphMLP, self).__init__()
        
        self.graph_encoder = MLP(graph_feat_dim, hidden_dim, dropout)
        self.hyper_encoder = MLP(hyper_feat_dim, hidden_dim, dropout)
        self.feature_encoder = None  # 

        self.similarity = Similarity(hidden_dim, activation_fn=torch.sigmoid)       
        self.fusion_layer = nn.Sequential(
                    nn.Linear(3 * hidden_dim, hidden_dim),
                    nn.GELU(),  # ,
                    nn.BatchNorm1d(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),  # ,
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(dropout),
                    )
        
        
        # 
        self.align_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.hidden_dim = hidden_dim
        
    def reset_parameters(self):

        # 
        for layer in self.graph_encoder.mlp_layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        
        # 
        for layer in self.hyper_encoder.mlp_layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        
        # 
        for layer in self.fusion_layer:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        
        # 
        for layer in self.align_projection:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        
        # 
        self.similarity.dense.reset_parameters()
    
    def encode_graph_view(self, x_graph, x_feat=None):
        # 
        if x_feat is not None and self.feature_encoder is not None:
            x_feat_encoded = self.feature_encoder(x_feat)
            x_graph = torch.cat([x_graph, x_feat_encoded], dim=-1)
        
        z_graph = self.graph_encoder(x_graph)
        return z_graph
    
    def encode_hyper_view(self, x_hyper, x_feat=None):
        if x_feat is not None and self.feature_encoder is not None:
            x_feat_encoded = self.feature_encoder(x_feat)
            x_hyper = torch.cat([x_hyper, x_feat_encoded], dim=-1)
        
        z_hyper = self.hyper_encoder(x_hyper)
        return z_hyper
    
    def encode_for_alignment(self, x_graph, x_hyper, x_feat=None):

        z_graph = self.encode_graph_view(x_graph, x_feat)
        z_hyper = self.encode_hyper_view(x_hyper, x_feat)
        
        # 
        z_graph_proj = self.align_projection(z_graph)
        z_hyper_proj = self.align_projection(z_hyper)
        
        return z_graph_proj, z_hyper_proj
    
    def encode_fused_view(self, x_graph, x_hyper, x_feat=None):

        z_graph = self.encode_graph_view(x_graph, x_feat)
        z_hyper = self.encode_hyper_view(x_hyper, x_feat)
        
        # 
        z_fused = torch.cat([x_graph, x_hyper], dim=-1)
        z_fused = self.fusion_layer(z_fused)
        
        return z_fused
    
    def similarity_from_embeddings(self, h_1, h_2):

        h = torch.stack((h_1, h_2), dim=1)  # [num_pairs, 2, hidden_dim]
        return self.similarity(h)
    
    def forward_single_view(self, x_pos_1, x_pos_2, x_feat_1=None, x_feat_2=None, view='graph'):

        if view == 'graph':
            encoder = self.encode_graph_view
        else:  # view == 'hyper'
            encoder = self.encode_hyper_view
        
        h_1 = encoder(x_pos_1, x_feat_1)
        h_2 = encoder(x_pos_2, x_feat_2)
        
        return self.similarity_from_embeddings(h_1, h_2)
    
    def forward_dual_view(self, x_graph_1, x_hyper_1, x_graph_2, x_hyper_2, x_feat_1=None, x_feat_2=None):
        """
        
        """
        # 
        h_1 = self.encode_fused_view(x_graph_1, x_hyper_1, x_feat_1)
        h_2 = self.encode_fused_view(x_graph_2, x_hyper_2, x_feat_2)
        
        return self.similarity_from_embeddings(h_1, h_2)

class ContrastiveLoss(nn.Module):

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, y_true, y_pred, sample_weights=None):

        y_true = y_true.float()
        # 
        pos_loss = y_true * torch.square(y_pred)
        neg_loss = (1.0 - y_true) * torch.square(torch.clamp(self.margin - y_pred, min=0.0))
        per_sample_loss = pos_loss + neg_loss  # [B]

        if sample_weights is not None:
            sample_weights = sample_weights.to(y_pred.device).float()
            loss_contrastive = (per_sample_loss * sample_weights).sum() / (sample_weights.sum() + 1e-8)
        else:
            loss_contrastive = per_sample_loss.mean()

        return loss_contrastive

import dhg
import torch
import networkx as nx
import numpy as np
import torch.nn as nn
import math
import torch.nn.functional as F
from einops import rearrange, repeat
from collections import defaultdict
from tqdm import tqdm

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
from mamba_ssm import Mamba

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

class LightGCNGlobalEncoder(nn.Module):
    def __init__(self, embed_dim:int, n_layers:int=3, use_edge_dropout:bool=False, keep_prob:float=0.8):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.use_edge_dropout = use_edge_dropout
        self.keep_prob = keep_prob

    @torch.no_grad()
    def _dropout_adj(self, adj: torch.sparse.FloatTensor, keep_prob: float):
        """
        Edge dropout on a sparse adjacency (coalesced).
        Reweight by 1/keep_prob to keep expectation.
        """
        if adj.layout != torch.sparse_coo:
            adj = adj.coalesce().to_sparse_coo()
        idx = adj.indices().t()          # [nnz, 2]
        val = adj.values()               # [nnz]
        mask = (torch.rand(val.size(0), device=val.device) < keep_prob)
        idx = idx[mask]
        val = val[mask] / keep_prob
        out = torch.sparse_coo_tensor(idx.t(), val, adj.size(), device=adj.device)
        return out.coalesce()

    def forward(self, adj_tilde: torch.sparse.FloatTensor, node_embeddings: torch.Tensor):
        """
        adj_tilde: symmetrically normalized adjacency (no self-loop), shape [N,N], sparse
        node_embeddings: initial node embeddings, shape [N, embed_dim]
        return: node embeddings after layer-combination, shape [N, embed_dim]
        """
        assert adj_tilde.is_sparse, "LightGCN expects a sparse normalized adjacency."
        
        # 
        device = node_embeddings.device
        adj_tilde = adj_tilde.to(device)
        E = node_embeddings  # 
        
        embs = [E]

        g = adj_tilde
        if self.use_edge_dropout and self.training:
            g = self._dropout_adj(adj_tilde, self.keep_prob)

        for _ in range(self.n_layers):
            E = torch.sparse.mm(g, E)                # E^(k+1) = A_tilde @ E^(k)
            embs.append(E)

        # layer combination: uniform mean over (K+1) layers
        embs = torch.stack(embs, dim=1)               # [N, K+1, D]
        E_final = embs.mean(dim=1)                    # [N, D]
        return E_final
import random
import os

class DNE:
    def __init__(self,
                 graph,
                 graph_test=None,
                 feat=None,
                 hidden_dim=128,
                 num_pos_features=256,
                 lambda_dual=1e-2,
                 lambda_pair_h=1e-2,
                 lambda_prior=1e-3):
        super().__init__()

        self.feat = feat                          # 
        self.hidden_dim = hidden_dim
        self.num_pos_features = num_pos_features  # 
        self.num_features = feat.shape[1] if feat is not None else None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.idx2node, self.node2idx = preprocess_nxgraph(graph)
        self.graph = nx.relabel_nodes(graph, self.node2idx)
        self.num_nodes = self.graph.number_of_nodes()

        self._embeddings = {}

        # Mamba 
        self.mamba = Mamba(d_model=self.num_pos_features,
                           d_state=32, d_conv=2, expand=1,
                           use_fast_path=False).to(self.device)

        # =========  =========
        self.H0 = None            #  incidence matrix [num_nodes, num_e]
        self.hyper_alpha = None   #  gating  [num_e]

        # 
        self.lambda_dual = lambda_dual     
        self.lambda_pair_h = lambda_pair_h  
        self.lambda_prior = lambda_prior   

    # ----------------------------------------------------------------------
    # 
    def _simulate_walks(self, graph, n, l, p, q, use_rejection_sampling=True):
        walker = RandomWalker(graph, p=p, q=q, use_rejection_sampling=use_rejection_sampling)
        walker.preprocess_transition_probs()
        walks = walker.simulate_walks(num_walks=n, walk_length=l, workers=1, verbose=1)
        return walks

    def _sample_hypergraph_node_walks_edrw(self,
                                        num_walks_per_node: int,
                                        walk_len: int,
                                        H: torch.Tensor,
                                        gamma,
                                        omega: torch.Tensor):
        """
        EDRW 
            P(e | v) ∝ ω(e) * γ_e(v)
            P(w | e) ∝ γ_e(w)
        :
            walks: list of [v_0, v_1, ..., v_{L-1}]
        """
        num_v, num_e = H.shape
        walks = []

        for v0 in range(num_v):
            incident_edges = self.v2e[v0]
            if len(incident_edges) == 0:
                for _ in range(num_walks_per_node):
                    walks.append([v0] * walk_len)
                continue

            for _ in range(num_walks_per_node):
                cur = v0
                path = [cur]
                for _ in range(walk_len - 1):
                    edges = self.v2e[cur]
                    if not edges:
                        path.append(cur)
                        continue

                    # --- Step 1: P(e | cur) ∝ ω(e) * γ_e(cur)
                    e_weights = []
                    for e in edges:
                        g_cur = gamma[e].get(cur, 0.0)
                        e_weights.append(float(omega[e].item()) * g_cur)
                    s = sum(e_weights)
                    if s <= 0:
                        probs_e = [1.0 / len(edges)] * len(edges)  # 
                    else:
                        probs_e = [w / s for w in e_weights]
                    e = random.choices(edges, weights=probs_e, k=1)[0]

                    # --- Step 2: P(w | e) ∝ γ_e(w)
                    nodes_in_e = self.e2v[e]
                    probs_w = [gamma[e].get(int(u), 0.0) for u in nodes_in_e]
                    s = sum(probs_w)
                    if s <= 0:
                        probs_w = [1.0 / len(nodes_in_e)] * len(nodes_in_e)
                    else:
                        probs_w = [w / s for w in probs_w]
                    nxt = random.choices(nodes_in_e, weights=probs_w, k=1)[0]

                    path.append(int(nxt))
                    cur = int(nxt)

                walks.append(path)

        return walks  # list[list[int]]
    def _build_clique_hypergraph(self):

        if getattr(self, "H0", None) is not None and \
        getattr(self, "v2e", None) is not None and \
        getattr(self, "e2v", None) is not None:
            return

        num_nodes = self.graph.number_of_nodes()
        cliques = list(nx.find_cliques(self.graph))
        e_list = [list(c) for c in cliques if len(c) >= 2]

        if len(e_list) == 0:
            raise RuntimeError("）")

        HG = dhg.Hypergraph(num_nodes, e_list)
        H0 = HG.H.to_dense().to(self.device)  # [N, E]
        num_v, num_e = H0.shape

        # 
        if getattr(self, "H0", None) is None or getattr(self, "hyper_alpha", None) is None:
            self._init_learnable_hypergraph(H0)   

        # v2e / e2v
        v2e = [[] for _ in range(num_v)]
        e2v = []
        for e_id, nodes in enumerate(e_list):
            e2v.append(list(nodes))
            for v in nodes:
                v2e[v].append(e_id)

        self.v2e = v2e
        self.e2v = e2v
        self.num_nodes_hg = num_v
        self.num_edges_hg = num_e


    def _build_gamma(self, H: torch.Tensor):

        num_v, num_e = H.shape
        gamma = []
        
        global_degrees = dict(self.graph.degree())
        
        for e in range(num_e):
            vs = self.e2v[e] 
            if len(vs) == 0:
                gamma.append({})
                continue
            

            subgraph = self.graph.subgraph(vs)
            
            # dict result: {node_id: degree_in_subgraph}
            local_deg_dict = dict(subgraph.degree(vs))
            
            local_degs = torch.tensor([local_deg_dict[v] for v in vs], dtype=torch.float32)
            
            if local_degs.max() == local_degs.min() and len(vs) > 1:
                weights = torch.tensor([global_degrees[v] for v in vs], dtype=torch.float32)
            else:
                # 
                weights = local_degs

            #
            # 
            total_w = weights.sum()
            
            if total_w.item() <= 1e-6:
                # 
                prob = torch.ones_like(weights) / len(vs)
            else:

                prob = weights / total_w

            # ---  ---
            
            gamma.append({int(v): float(p) for v, p in zip(vs, prob)})
        
        return gamma


    def _compute_edge_weights(self, H: torch.Tensor, node_feat_for_omega: torch.Tensor = None):
        """
        

            ω*(e) = <S, A^(e)> / (||A^(e)||_F^2 + λ_ω |e|^{β_ω})
                = (sum_{u,v in e} sim(z_u, z_v)) / (|e|^2 + λ_ω |e|^{β_ω})

         A^(e)_{uv}=1 iff u, ||A^(e)||_F^2 = |e|^2。
        """
        num_v, num_e = H.shape
        device = H.device

        if node_feat_for_omega is None:
            # raise ValueError("node_feat_for_omega is required to compute ω(e) consistent with the paper.")
            omega = torch.ones(num_e, dtype=torch.float32, device=device)
            return omega.clamp(min=1e-6)

        lambda_omega = float(getattr(self, "lambda_omega", 0.2))
        beta_omega   = float(getattr(self, "beta_omega", 0.2))
        eps          = 1e-12

        z = node_feat_for_omega.to(device)              # [N, D]
        z = F.normalize(z, dim=-1)                      # cosine 

        omega_list = []

        for e in range(num_e):
            vs = self.e2v[e]
            m = len(vs)

            if m == 0:
                omega_list.append(torch.tensor(0.0, device=device))
                continue

            # m==1 {u,v in e} S_uv = S_vv = 1（cosine）
            if m == 1:
                numerator = torch.tensor(1.0, device=device)
            else:
                z_e = z[vs]                             # [m, D]
                # cosine similarity matrix: sim_{ij} = <z_i, z_j>
                sim = z_e @ z_e.t()                     # [m, m]
                numerator = sim.sum()                   # sum_{u,v in e} 

            denom = (m * m) + (lambda_omega * (m ** beta_omega))
            omega_e = numerator / (denom + eps)

            
            omega_list.append(torch.clamp(omega_e, min=0.0))

        omega = torch.stack(omega_list).to(torch.float32)

       
        omega = omega / omega.mean().clamp(min=1e-6)

        return omega.clamp(min=1e-6)

    def prepare_data_with_hypergraph(
        self,
        n,
        l,
        p,
        q,
        x_pos=None,
        z_struct=None,
        strong_weight: float = 2.0,   
        weak_weight: float = 0.5,     
        hyper_walk_ratio: float = 0.5,
        hard_neg_topk: int =300,     
        hard_neg_exclude_topm: int = 30,  
        neg_ratio: float = 2.0,       
        hard_frac: float = 0.3,
    ):

        import math
        import random
        from collections import defaultdict
        import numpy as np
        import torch
        import torch.nn.functional as F

        device = self.device

        # ======================
        # ======================
        degrees = defaultdict(int, dict(self.graph.degree()))
        all_nodes = list(degrees.keys())
        sampling_distribution = np.array(
            [degrees[node_id] ** 0.75 for node_id in all_nodes],
            dtype=np.float32
        )
        sampling_distribution_norm = sampling_distribution / sampling_distribution.sum()

        #  neg_ratio
        if neg_ratio < 0:
            raise ValueError(f"neg_ratio must be >= 0, got {neg_ratio}")
        neg_ratio_int = int(math.floor(neg_ratio))
        neg_ratio_frac = float(neg_ratio - neg_ratio_int)
        hard_frac = max(0.0, min(1.0, hard_frac))

        num_nodes = self.graph.number_of_nodes()

        # ======================
        # ======================
        walks_original = self._simulate_walks(self.graph, n=n, l=l, p=p, q=q)

        targets_original = [walk[0] for walk in walks_original]
        positive_pairs_original = torch.tensor(
            [
                (target, context)
                for target, walk in zip(targets_original, walks_original)
                for context in walk[1:]
            ],
            dtype=torch.long,
            device=device,
        )


        self._build_clique_hypergraph()         
        H_hg = self.H0.to(device)                

        positive_pairs_hyper = torch.empty((0, 2), dtype=torch.long, device=device)

        if hyper_walk_ratio > 0 and H_hg.numel() > 0:

            gamma = self._build_gamma(H_hg)   # list[dict]，gamma[e][v] = γ_e(v)

            node_feat_for_omega = z_struct.to(device) if z_struct is not None else None
            omega = self._compute_edge_weights(H_hg, node_feat_for_omega=node_feat_for_omega)  # [E]


            num_walks_per_node = max(1, int(n * hyper_walk_ratio))

            valid_start_nodes = [v for v in range(num_nodes) if len(self.v2e[v]) > 0]

            def _hypergraph_random_walk_edrw(start_v, walk_len, rng=None):
                """
                EDRW:  P(e | v) ∝ ω(e) * γ_e(v) ，P(w | e) ∝ γ_e(w)
                """
                if rng is None:
                    rng = np.random

                walk = [start_v]
                current = start_v
                for _ in range(walk_len - 1):
                    edges = self.v2e[current]
                    if not edges:
                        
                        walk.append(current)
                        continue

                    # Step 1: P(e | current) ∝ ω(e) * γ_e(current)
                    e_weights = []
                    for e in edges:
                        g_cur = gamma[e].get(current, 0.0)  # γ_e(current)
                        e_weights.append(float(omega[e].item()) * g_cur)
                    s = sum(e_weights)
                    if s <= 0:
                       
                        e_weights = [float(omega[e].item()) for e in edges]
                        s2 = sum(e_weights)
                        if s2 <= 0:
                            probs_e = [1.0 / len(edges)] * len(edges)
                        else:
                            probs_e = [w / s2 for w in e_weights]
                    else:
                        probs_e = [w / s for w in e_weights]

                    e_id = rng.choice(edges, p=probs_e)

                    # Step 2: P(w | e) ∝ γ_e(w)
                    nodes_in_e = self.e2v[e_id]
                    if not nodes_in_e:
                        walk.append(current)
                        continue

                    probs_w = [gamma[e_id].get(int(v), 0.0) for v in nodes_in_e]
                    s_w = sum(probs_w)
                    if s_w <= 0:
                        probs_w = [1.0 / len(nodes_in_e)] * len(nodes_in_e)
                    else:
                        probs_w = [w / s_w for w in probs_w]

                    nxt = int(rng.choice(nodes_in_e, p=probs_w))
                    walk.append(nxt)
                    current = nxt

                return walk

            
            hyper_walks = []
            if len(valid_start_nodes) > 0:
                for _ in range(num_walks_per_node):
                    for v in valid_start_nodes:
                        w = _hypergraph_random_walk_edrw(v, l)
                        if len(w) > 1:
                            hyper_walks.append(w)

           
            pos_pairs_hyper = []
            for walk in hyper_walks:
                target = walk[0]
                for context in walk[1:]:
                    if target != context:
                        pos_pairs_hyper.append([target, context])

            if len(pos_pairs_hyper) > 0:
                positive_pairs_hyper = torch.tensor(
                    pos_pairs_hyper, dtype=torch.long, device=device
                )

        # ======================
        
        # ======================
        positive_pairs = torch.cat(
            [positive_pairs_original, positive_pairs_hyper], dim=0
        )  # [num_pos, 2]

        num_pos = positive_pairs.size(0)
        num_pos_original = positive_pairs_original.size(0)
        num_pos_hyper = positive_pairs_hyper.size(0)

        labels_pos = torch.ones(num_pos, dtype=torch.long, device=device)

        # ======================
        
        # ======================
        N = num_nodes  

        
        pos_neighbors = [set() for _ in range(N)]
        for idx in range(num_pos):
            i = int(positive_pairs[idx, 0].item())
            j = int(positive_pairs[idx, 1].item())
            if 0 <= i < N and 0 <= j < N:
                pos_neighbors[i].add(j)
                pos_neighbors[j].add(i)

        # ----------------------
       
        # ----------------------
        if (x_pos is None) or (z_struct is None):
            negative_pairs = []
            for idx in range(num_pos):
                i = int(positive_pairs[idx, 0].item())
                
                k = neg_ratio_int
                if random.random() < neg_ratio_frac:
                    k += 1
                for _ in range(k):
                    j = int(
                        np.random.choice(all_nodes, size=1, p=sampling_distribution_norm)[0]
                    )
                    if i == j:
                        continue
                    negative_pairs.append([i, j])

            if len(negative_pairs) == 0:
                node_pairs = positive_pairs
                labels = labels_pos
                sample_weights = torch.ones(len(node_pairs), dtype=torch.float32, device=device)

                print(
                    f"[prepare_data_with_hypergraph-fallback] #pos_total={num_pos}, "
                    f"no negatives generated (neg_ratio={neg_ratio})"
                )
                return node_pairs, labels, sample_weights

            negative_pairs = torch.tensor(negative_pairs, dtype=torch.long, device=device)
            num_neg = negative_pairs.size(0)
            labels_neg = torch.zeros(num_neg, dtype=torch.long, device=device)

            node_pairs = torch.cat([positive_pairs, negative_pairs], dim=0)
            labels = torch.cat([labels_pos, labels_neg], dim=0)
            sample_weights = torch.ones(len(node_pairs), dtype=torch.float32, device=device)

            indices = torch.randperm(len(node_pairs), device=device)
            node_pairs = node_pairs[indices]
            labels = labels[indices]
            sample_weights = sample_weights[indices]

            print(
                f"[prepare_data_with_hypergraph-fallback] #pos_total={num_pos}, "
                f"#original={num_pos_original}, #hyper={num_pos_hyper}, "
                f"#neg={num_neg}, neg_ratio≈{num_neg / max(1, num_pos):.3f}"
            )
            return node_pairs, labels, sample_weights

        # ----------------------
        
        # ----------------------
        with torch.no_grad():
            x_pos = x_pos.to(device)
            z_struct = z_struct.to(device)

            # x_pos_norm = x_pos
            # z_struct_norm = z_struct
            x_pos_norm = F.normalize(x_pos, p=2, dim=1)        # [N, D1]
            z_struct_norm = F.normalize(z_struct, p=2, dim=1)  # [N, D2]
            S_pos = torch.matmul(x_pos_norm, x_pos_norm.t())          # [N, N]
            S_struct = torch.matmul(z_struct_norm, z_struct_norm.t()) # [N, N]
            S_comb = 0.5 * (S_pos + S_struct)                         # [N, N]

            easy_neg_candidates = [[] for _ in range(N)]
            hard_neg_candidates = [[] for _ in range(N)]

            for i in range(N):
                sim_row = S_comb[i]  # [N]
                sim_row_i = sim_row.clone()
                sim_row_i[i] = -1e9   

                _, idx_sorted = torch.sort(sim_row_i, dim=0, descending=False)
                idx_sorted = idx_sorted.cpu().numpy().tolist()

                
                idx_sorted_no_pos = [j for j in idx_sorted if j not in pos_neighbors[i]]
                if len(idx_sorted_no_pos) == 0:
                    continue

              
                num_easy_pool = max(1, min(len(idx_sorted_no_pos) // 3, 100))
                easy_candidates = idx_sorted_no_pos[:num_easy_pool]
                
                idx_sorted_desc = list(reversed(idx_sorted_no_pos)) 
                start = hard_neg_exclude_topm
                end = min(start + hard_neg_topk, len(idx_sorted_desc))
                if start < end:
                    hard_candidates = idx_sorted_desc[start:end]
                else:
                    hard_candidates = []

                easy_neg_candidates[i] = easy_candidates
                hard_neg_candidates[i] = hard_candidates

        def _sample_struct_neg_for_node(i, prefer_hard: bool):
            #  hard
            if prefer_hard:
                if len(hard_neg_candidates[i]) > 0:
                    j = random.choice(hard_neg_candidates[i])
                    return j, True
                if len(easy_neg_candidates[i]) > 0:
                    j = random.choice(easy_neg_candidates[i])
                    return j, False
            #  easy
            else:
                if len(easy_neg_candidates[i]) > 0:
                    j = random.choice(easy_neg_candidates[i])
                    return j, False
                if len(hard_neg_candidates[i]) > 0:
                    j = random.choice(hard_neg_candidates[i])
                    return j, True

            
            j = int(
                np.random.choice(all_nodes, size=1, p=sampling_distribution_norm)[0]
            )
            return j, False 

        
        negative_pairs = []
        neg_is_hard_flags = []

        for idx in range(num_pos):
            i = int(positive_pairs[idx, 0].item())

            k = neg_ratio_int
            if random.random() < neg_ratio_frac:
                k += 1
            if k <= 0:
                continue

            for _ in range(k):
                prefer_hard = (random.random() < hard_frac)
                j, is_hard = _sample_struct_neg_for_node(i, prefer_hard=prefer_hard)
                if i == j:
                    continue
                negative_pairs.append([i, j])
                neg_is_hard_flags.append(is_hard)

        if len(negative_pairs) == 0:
            negative_samples = np.random.choice(
                all_nodes, size=num_pos, p=sampling_distribution_norm
            )
            negative_samples = torch.from_numpy(negative_samples).long().to(device)
            negative_pairs = torch.stack(
                (positive_pairs[:, 0], negative_samples), dim=1
            )
            neg_is_hard_flags = [False] * num_pos
        else:
            negative_pairs = torch.tensor(
                negative_pairs, dtype=torch.long, device=device
            )

        num_neg = negative_pairs.size(0)
        neg_is_hard = torch.tensor(
            neg_is_hard_flags, dtype=torch.bool, device=device
        )

        labels_neg = torch.zeros(num_neg, dtype=torch.long, device=device)

        
        node_pairs = torch.cat([positive_pairs, negative_pairs], dim=0)
        labels = torch.cat([labels_pos, labels_neg], dim=0)

        # ======================
       
        # ======================
        sample_weights = torch.ones(len(node_pairs), dtype=torch.float32, device=device)

        
        sample_weights[:num_pos] = 1.0

        
        sample_weights[num_pos:][neg_is_hard] = strong_weight  # hard neg
        sample_weights[num_pos:][~neg_is_hard] = weak_weight   # easy neg

        # ======================
        
        # ======================
        indices = torch.randperm(len(node_pairs), device=device)
        node_pairs = node_pairs[indices]
        labels = labels[indices]
        sample_weights = sample_weights[indices]

        num_neg_hard = int(neg_is_hard.sum().item())
        num_neg_easy = int(num_neg - num_neg_hard)

        print(
            f"[prepare_data_with_hypergraph] #pos_total={num_pos} "
            f"(original={num_pos_original}, hyper={num_pos_hyper}), "
            f"#neg_total={num_neg} (easy={num_neg_easy}, hard={num_neg_hard}), "
            f"neg_ratio≈{num_neg / max(1, num_pos):.3f}, "
            f"hard_w={strong_weight}, easy_w={weak_weight}"
        )

        return node_pairs, labels, sample_weights

    def _init_learnable_hypergraph(self, H0: torch.Tensor):

        self.H0 = H0.detach() 
        num_e = self.H0.shape[1]
        
        self.hyper_alpha = nn.Parameter(torch.zeros(num_e, device=self.device))

    def get_learnable_H(self):

        assert self.H0 is not None and self.hyper_alpha is not None, "Hypergraph not initialized."

        
        weights = F.softplus(self.hyper_alpha)  # [num_e]
        H = self.H0 * weights  # [num_nodes, num_e] * [num_e] 
        
        H_row_sum = H.sum(dim=1, keepdim=True).clamp(min=1e-6)
        H_norm = H / H_row_sum
        return H

    def hypergraph_propagate(self, node_emb: torch.Tensor):

        H = self.get_learnable_H()      # [N, E]
        hyper_emb = torch.matmul(H.t(), node_emb)  # [E, d]  
        hyper_node = torch.matmul(H, hyper_emb)    # [N, d]  
        return hyper_node

    def get_hypergraph_enhanced_pos(self, base_x_pos: torch.Tensor):

        hyper_node = self.hypergraph_propagate(base_x_pos)
        # x_pos_dyn = base_x_pos + hyper_node
        x_pos_dyn = hyper_node
        return x_pos_dyn
    def compute_pos_embLE(self, graph,num_pos_features=256):
        pos_emb_model = LE(graph, num_pos_features)# 
        x_pos = pos_emb_model._X#
        return x_pos
    # ----------------------------------------------------------------------
    
    import numpy as np

    def compute_pos_emb(self, num_pos_features: int = 256):

        import random
        d_model = num_pos_features
        walk_len = 10
        num_walks = 10

        # =====================================================
        
        # =====================================================
        cliques = list(nx.find_cliques(self.graph))
        e_list = [list(clique) for clique in cliques if len(clique) >= 2]

        num_nodes = self.graph.number_of_nodes()
        HG = dhg.Hypergraph(num_nodes, e_list)

       
        H0 = HG.H.to_dense().to(self.device)  # [V, E]
        num_v, num_e = H0.shape
        
        if getattr(self, "H0", None) is None or getattr(self, "hyper_alpha", None) is None:
            self._init_learnable_hypergraph(H0)

        
        H = self.H0.to(self.device) if hasattr(self, "H0") else H0  # [V, E]

        #
        v2e = []
        for v in range(num_v):
            
            incident_e = torch.nonzero(H[v, :] > 0, as_tuple=False).view(-1)
            v2e.append(incident_e.tolist())

        
        e2v = []
        for e in range(num_e):
            incident_v = torch.nonzero(H[:, e] > 0, as_tuple=False).view(-1)
            e2v.append(incident_v.tolist())

        gamma = []
        for e in range(num_e):
            vs = e2v[e]
            if not vs:
                gamma.append({})
                continue
            vals = H[vs, e]  # [|e|]
            s = vals.sum()
            if s.item() <= 0:
                
                prob = torch.ones_like(vals) / len(vs)
            else:
                prob = vals / s
            gamma.append({int(v): float(p) for v, p in zip(vs, prob)})

        omega = torch.ones(num_e, device=self.device)

        # =====================================================

        # =====================================================
        def hypergraph_node_walks(num_walks_per_node: int, walk_len: int):

            walks = []
            for v in range(num_v):
                incident_e = v2e[v]
                
                if len(incident_e) == 0:
                    for _ in range(num_walks_per_node):
                        walks.append([v] * walk_len)
                    continue

                for _ in range(num_walks_per_node):
                    cur = v
                    path = [cur]
                    for _ in range(walk_len - 1):
                        incident_e = v2e[cur]
                        if not incident_e:
                            
                            path.append(cur)
                            continue

                        
                        e = random.choice(incident_e)

                        nodes_in_e = e2v[e]
                        if not nodes_in_e:
                            path.append(cur)
                            continue

                        probs = [gamma[e].get(int(v_id), 0.0) for v_id in nodes_in_e]
                        s = sum(probs)
                        if s <= 0:
                            probs = [1.0 / len(nodes_in_e)] * len(nodes_in_e)
                        else:
                            probs = [p / s for p in probs]

                        nxt = random.choices(nodes_in_e, weights=probs, k=1)[0]
                        path.append(int(nxt))
                        cur = int(nxt)

                    walks.append(path)
            return walks

        node_walks_list = hypergraph_node_walks(num_walks_per_node=num_walks, walk_len=walk_len)

        padded_node_walks = []
        for walk in node_walks_list:
            if len(walk) < walk_len:
                walk = walk + [walk[-1]] * (walk_len - len(walk))
            elif len(walk) > walk_len:
                walk = walk[:walk_len]
            padded_node_walks.append(walk)
        node_walks = torch.tensor(padded_node_walks, device=self.device, dtype=torch.long)

        L = torch.mm(H.t(), H)  # [E, E]
        L.fill_diagonal_(0)
        L_edge_index = (L > 0).nonzero(as_tuple=False).t().to(self.device)

        L_graph = nx.Graph()
        L_graph.add_nodes_from(range(num_e))
        L_graph.add_edges_from(L_edge_index.t().cpu().numpy())

        hyper_walks_list = self._simulate_walks(
            L_graph, n=num_walks, l=walk_len - 1, p=1, q=1, use_rejection_sampling=True
        )
        padded_hyper_walks = []
        for walk in hyper_walks_list:
            if len(walk) < walk_len:
                walk = walk + [walk[-1]] * (walk_len - len(walk))
            elif len(walk) > walk_len:
                walk = walk[:walk_len]
            padded_hyper_walks.append(walk)
        hyper_walks = torch.tensor(padded_hyper_walks, device=self.device, dtype=torch.long)

        node_start_ids = torch.arange(num_nodes, device=self.device).repeat_interleave(num_walks)
        hyper_start_ids = torch.arange(num_e, device=self.device).repeat(num_walks)
        # =====================================================
        # =====================================================
        node_emb = nn.Embedding(num_nodes, d_model).to(self.device)
        hyper_emb = nn.Embedding(num_e, d_model).to(self.device)

        node_hidden = node_emb(node_walks)    # [B_node, L, D]
        hyper_hidden = hyper_emb(hyper_walks) # [B_hyper, L, D]

        node_out = self.mamba(node_hidden)    # [B_node, L, D]
        hyper_out = self.mamba(hyper_hidden)  # [B_hyper, L, D]

        node_emb_from_seq = torch.zeros(num_nodes, d_model, device=self.device)
        counts = torch.zeros(num_nodes, device=self.device)
        for i in range(node_out.shape[0]):
            start = node_start_ids[i]
            seq_mean = node_out[i].mean(dim=0)  # [D]
            node_emb_from_seq[start] += seq_mean
            counts[start] += 1
        node_emb_from_seq /= counts.unsqueeze(1).clamp(min=1)

        hyper_emb_from_seq = torch.zeros(num_e, d_model, device=self.device)
        hyper_counts = torch.zeros(num_e, device=self.device)
        for i in range(hyper_out.shape[0]):
            start = hyper_start_ids[i]
            seq_mean = hyper_out[i].mean(dim=0)  # [D]
            hyper_emb_from_seq[start] += seq_mean
            hyper_counts[start] += 1
        hyper_emb_from_seq /= hyper_counts.unsqueeze(1).clamp(min=1)

        H_row_sum = H.sum(dim=1, keepdim=True).clamp(min=1)
        node_emb_from_hyper = torch.mm(H, hyper_emb_from_seq) / H_row_sum

        final_node_emb = node_emb_from_hyper
        x_pos_node = node_emb_from_seq.detach().cpu().numpy()
        x_pos_hyper = node_emb_from_hyper.detach().cpu().numpy()
        return x_pos_node, x_pos_hyper


    def build_adjacency_from_nx_graph(self, graph):
        num_nodes = len(graph.nodes())
        
        try:
            adj_sparse = nx.adjacency_matrix(graph)
            
            adj = adj_sparse.toarray().astype(np.float32)
            
            print(f"node num={num_nodes}, edge={graph.number_of_edges()}")
            
        except Exception as e:
            print(f": {e}")
            adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
            
            for i, j in graph.edges():
                adj[i, j] = 1.0
                if not graph.is_directed():
                    adj[j, i] = 1.0
        
        np.fill_diagonal(adj, 1.0)
        rowsum = np.array(adj.sum(1))
        r_inv_sqrt = np.power(rowsum, -0.5).flatten()
        r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
        r_mat_inv_sqrt = np.diag(r_inv_sqrt)
        adj_normalized = r_mat_inv_sqrt @ adj @ r_mat_inv_sqrt
        
        return torch.FloatTensor(adj_normalized).to(self.device)
    def normalize_adj_matrix(self,adj: torch.Tensor, symmetric: bool=True):

       
        degree = adj.sum(dim=1).cpu().numpy() 
        degree_inv_sqrt = 1. / torch.sqrt(torch.tensor(degree, dtype=torch.float32)).to(self.device)  # D^(-1/2)
        degree_inv_sqrt_matrix = torch.diag(degree_inv_sqrt).to(self.device)
        adj=adj.to(self.device)
        
        if symmetric:
            adj_normalized = degree_inv_sqrt_matrix @ adj @ degree_inv_sqrt_matrix  # D^(-1/2) * A * D^(-1/2)
        else:
            adj_normalized = adj * degree_inv_sqrt_matrix  # D^(-1) * A
        
       
        adj_normalized = adj_normalized.to_sparse()
        return adj_normalized
    def _ensure_hypergraph_initialized(self):

        if getattr(self, "H0", None) is not None and getattr(self, "hyper_alpha", None) is not None:
            return 

       
        cliques = list(nx.find_cliques(self.graph))
        e_list = [list(c) for c in cliques if len(c) >= 2]

        num_nodes = self.graph.number_of_nodes()
        if len(e_list) == 0:
            
            raise RuntimeError("No clique hyperedges found when initializing hypergraph.")

        import dhg
        HG = dhg.Hypergraph(num_nodes, e_list)
        H0 = HG.H.to_dense().to(self.device)   # [N, E]

       
        self._init_learnable_hypergraph(H0)

        print(f"[Hypergraph] initialized: H0 shape={self.H0.shape}, "
            f"num_nodes={num_nodes}, num_edges={self.H0.shape[1]}")

    # ----------------------------------------------------------------------
    def train(self, batch_size=1000, epochs=5, walk_number=50, walk_length=10, p=1.0, q=1.0):

        
        save_dir = "../dne_retrain"
        # mamba_ckpt = os.path.join(save_dir, "mamba_hyper_pretraincora.pt")
        x_mamba_path = os.path.join(save_dir, "x_posMamb_atha.npy")
        x_hmamba_path = os.path.join(save_dir, "x_posHMamb_atha.npy")
        self.x_posMamba = np.load(x_mamba_path)
        self.x_posHMamba = np.load(x_hmamba_path)
        self.x_posLE = self.compute_pos_embLE(self.graph,num_pos_features=self.num_pos_features)
        print("[train] loaded pretrained Mamba encoder & embeddings.")
        self.base_x_pos = torch.from_numpy(self.x_posHMamba).float().to(self.device)  # [N, D]
        x_pos = torch.from_numpy(self.x_posLE).float().to(self.device)  
        self._ensure_hypergraph_initialized()

        if self.num_features is not None:
            x_feat = torch.from_numpy(self.feat).float().to(self.device)
        else:
            x_feat = None
        full_adj_original = self.build_adjacency_from_nx_graph(self.graph)
        adj_original_sparse = self.normalize_adj_matrix(full_adj_original) 
        self.gcnii_original = LightGCNGlobalEncoder(self.hidden_dim, 
                                            n_layers=1,
                                            use_edge_dropout=False,
                                            keep_prob=0.7).to(self.device)

        z_struct_original = self.gcnii_original(adj_original_sparse,x_pos) 
        # node_pairs, labels,sample_weights = self.prepare_data(walk_number, walk_length, p, q)
        node_pairs, labels, sample_weights = self.prepare_data_with_hypergraph(
                n=walk_number,
                l=walk_length, 
                p=p,
                q=q,
                x_pos=x_pos,
                z_struct=z_struct_original,
                strong_weight=2,
                weak_weight=1,
                hyper_walk_ratio=2#
            )
        node_pairs = node_pairs.to(self.device)
        labels = labels.to(self.device)
        graph_feat_dim = self.num_pos_features + z_struct_original.shape[1]
        hyper_feat_dim = self.num_pos_features

        # ==================== 2.  ====================
        model = DualViewGraphMLP(
            graph_feat_dim=graph_feat_dim,
            hyper_feat_dim=hyper_feat_dim,
            hidden_dim=self.hidden_dim,
            dropout=0.1
        ).to(self.device)

        criterion = ContrastiveLoss().to(self.device)

        params = list(model.parameters())
        if self.hyper_alpha is not None:
            params.append(self.hyper_alpha)
        optimizer = torch.optim.Adam(params, lr=1e-3)

        n_batch = (len(node_pairs) + batch_size - 1) // batch_size
        epoch_iter = tqdm(range(epochs), desc="Training Epochs")
        # ==================== 4.  ====================
        tau = 0.7  # 
        lambda_align = 0.2  # 
        lambda_pair_h = 0.05  # 
        lambda_prior = 0.05  # 
        for epoch in epoch_iter:
            model.train()
            total_loss = 0.0
            total_main = 0.0
            total_align = 0.0
            total_pair_h = 0.0
            total_prior = 0.0
            
            for i in range(n_batch):
                start = i * batch_size
                end = min((i + 1) * batch_size, len(node_pairs))
                batch_node_pairs = node_pairs[start:end]     # [B, 2]
                batch_labels = labels[start:end]             # [B]
                batch_sample_weights = sample_weights[start:end]
                optimizer.zero_grad()

                # ==================== 5.  ====================
                x_pos_dyn = self.get_hypergraph_enhanced_pos(self.base_x_pos)  # [N, D]
                # ==================== 6.  ====================
                # x_graph = torch.cat([x_pos, z_struct_original], dim=-1)  # [N, D_graph]
                x_hyper = x_pos_dyn # [N, D_hyper]
                x_graph = torch.cat([x_pos, z_struct_original], dim=-1) 
                # ==================== 7.====================
                nodes_in_batch = torch.unique(batch_node_pairs.flatten())  # [M]
                idx_i = batch_node_pairs[:, 0]
                idx_j = batch_node_pairs[:, 1]
                
                # 
                z_graph_align, z_hyper_align = model.encode_for_alignment(
                    x_graph[nodes_in_batch],
                    x_hyper[nodes_in_batch],
                    x_feat[nodes_in_batch] if x_feat is not None else None
                )  # [M, hidden_dim], [M, hidden_dim]
                # 
                if x_feat is not None:
                    # 
                    h_i = model.encode_fused_view(
                        x_graph[idx_i], 
                        x_hyper[idx_i], 
                        x_feat[idx_i]
                    )
                    # 
                    h_j = model.encode_fused_view(
                        x_graph[idx_j], 
                        x_hyper[idx_j], 
                        x_feat[idx_j]
                    )
                else:
                    h_i = model.encode_fused_view(
                        x_graph[idx_i], 
                        x_hyper[idx_i]
                    )
                    # 
                    h_j = model.encode_fused_view(
                        x_graph[idx_j], 
                        x_hyper[idx_j]
                    )
                
                # ==================== 11.  ====================
                out = model.similarity_from_embeddings(h_i, h_j).squeeze()  # [B]
                loss_main = criterion(batch_labels, out, batch_sample_weights)
                # ==================== 10.   InfoNCE） ====================
                # 
                pos_mask_pairs = (batch_labels > 0)

                if pos_mask_pairs.any():
                    nodes_align = torch.unique(batch_node_pairs[pos_mask_pairs].flatten())  # [M]
                else:
                    nodes_align = None

                if nodes_align is None or nodes_align.numel() < 2:
                    loss_align = torch.zeros((), device=self.device)
                else:
                    # 
                    xg = x_graph[nodes_align]         # [M, Dg]
                    xh = x_hyper[nodes_align]         # [M, Dh]
                    xg_zero = torch.zeros_like(xg)
                    xh_zero = torch.zeros_like(xh)

                    if x_feat is not None:
                        xf = x_feat[nodes_align]
                        h_graph = model.encode_fused_view(xg,      xh_zero, xf)  # graph-only
                        h_hyper = model.encode_fused_view(xg_zero, xh,      xf)  # hyper-only
                    else:
                        h_graph = model.encode_fused_view(xg,      xh_zero)
                        h_hyper = model.encode_fused_view(xg_zero, xh)

                    #  InfoNCE


                    logits = (h_graph @ h_hyper.t()) / tau      # [M, M]
                    targets = torch.arange(logits.size(0), device=logits.device)

                    loss_align = 0.5 * (
                        F.cross_entropy(logits, targets) +
                        F.cross_entropy(logits.t(), targets)
                    )
                # ==================== 12.  ====================
                # 
                H = self.get_learnable_H()      # [N, E]  H_norm
                H_i = H[idx_i]                  # [B, E]
                H_j = H[idx_j]                  # [B, E]
                # 
                s_H = (H_i * H_j).sum(dim=1)    # [B]

                # 
                pos_mask = (batch_labels > 0)   # [B] bool

                if pos_mask.any():
                    logits_pos = s_H[pos_mask]  # [B_pos]
                    target_pos = torch.ones_like(logits_pos)

                    # element-wise BCEWithLogits  sigmoid）
                    per_loss = F.binary_cross_entropy_with_logits(
                        logits_pos, target_pos, reduction="none"
                    )
                    #  
                    w_pos = batch_sample_weights[pos_mask].to(per_loss.device).float()
                    loss_pair_h = (per_loss * w_pos).sum() / (w_pos.sum().clamp(min=1e-6))
                else:
                    loss_pair_h = torch.zeros((), device=self.device)
                # 
                weights = F.softplus(self.hyper_alpha)
                loss_prior = torch.mean((weights - 1.0) ** 2)

                loss = (loss_main 
                        + lambda_align * loss_align
                        + lambda_pair_h * loss_pair_h
                        + lambda_prior * loss_prior)
                
                # 
                loss.backward()
                
                # 
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # 
                optimizer.step()
                
                # ==================== 14.  ====================
                total_loss += loss.item()
                total_main += loss_main.item()
                # total_align += loss_align.item()
                total_pair_h += loss_pair_h.item()
                total_prior += loss_prior.item()
                
                # 
                epoch_iter.set_postfix({
                    "L": f"{loss.item():.4f}",
                    "L_main": f"{loss_main.item():.4f}",
                    # "L_align": f"{loss_align.item():.4f}",
                    "L_pair_h": f"{loss_pair_h.item():.4f}",
                    "L_prior": f"{loss_prior.item():.4f}",
                })
            
            # ==================== 15.  ====================
            avg_loss = total_loss / n_batch
            avg_main = total_main / n_batch
            avg_align = total_align / n_batch
            avg_pair_h = total_pair_h / n_batch
            avg_prior = total_prior / n_batch
            
            print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Main={avg_main:.4f}, "
                f"Align={avg_align:.4f}, Pair_H={avg_pair_h:.4f}, Prior={avg_prior:.4f}")

        # ==================== 6.  ====================
        print("Training completed. Saving model and variables...")
        # 
        self.model = model
        # 
        self.x_pos_ = x_pos  # 
        self.z_struct_original_ = z_struct_original  # 
        # 
        with torch.no_grad():
            self.x_pos_dyn_ = self.get_hypergraph_enhanced_pos(self.base_x_pos)
        # 
        self.x_graph_ = torch.cat([self.x_pos_, self.z_struct_original_], dim=-1)
        self.x_hyper_ = self.x_pos_dyn_
        # 
        if x_feat is not None:
            self.x_feat_ = x_feat
        # 
        with torch.no_grad():
            self.H_final_ = self.get_learnable_H()
        
        if self.hyper_alpha is not None:
            self.hyper_alpha_final_ = self.hyper_alpha.clone()
        
        print(f"Variables saved:")
        print(f"  - x_graph shape: {self.x_graph_.shape}")
        print(f"  - x_hyper shape: {self.x_hyper_.shape}")
        print(f"  - H_final shape: {self.H_final_.shape if hasattr(self, 'H_final_') else 'N/A'}")

    
    def get_embeddings(self, view_type='both'):
        if not hasattr(self, 'model'):
            raise ValueError("Model not trained yet. Please run training first.")
        self.model.eval()
        if not hasattr(self, 'x_graph_') or not hasattr(self, 'x_hyper_'):
            raise ValueError("Training variables not found. Please run training first.")
        x_graph = self.x_graph_
        x_hyper = self.x_hyper_
        # 
        x_feat = None
        if hasattr(self, 'x_feat_'):
            x_feat = self.x_feat_
        with torch.no_grad():
            #
            if view_type == 'graph':
                embeddings = self.model.encode_graph_view(x_graph, x_feat)           
            elif view_type == 'hyper':
                # 
                embeddings = self.model.encode_hyper_view(x_hyper, x_feat)
            elif view_type == 'fused':
                # 
                embeddings = self.model.encode_fused_view(x_graph, x_hyper, x_feat)
            elif view_type == 'both':
                # embeddings_fused = self.model.encode_fused_view(x_graph, x_hyper, x_feat)
                embeddings_graph = self.model.encode_fused_view(x_graph, x_hyper, x_feat)
                embeddings_hyper = self.model.encode_fused_view(x_graph, x_hyper, x_feat)
                embeddings_fused = self.model.encode_fused_view(x_graph, x_hyper, x_feat)
                #
                embeddings_graph = embeddings_graph.detach().cpu().numpy()
                embeddings_hyper = embeddings_hyper.detach().cpu().numpy()
                embeddings_fused = embeddings_fused.detach().cpu().numpy()
                
                # 
                idx2node = self.idx2node
                embeddings_dict_graph = {}
                embeddings_dict_hyper = {}
                embeddings_dict_fused = {}
                
                for i in range(len(idx2node)):
                    node_id = idx2node[i]
                    embeddings_dict_graph[node_id] = embeddings_graph[i]
                    embeddings_dict_hyper[node_id] = embeddings_hyper[i]
                    embeddings_dict_fused[node_id] = embeddings_fused[i]
                
                return embeddings_dict_graph, embeddings_dict_hyper, embeddings_dict_fused
            
            else:
                raise ValueError(f"Unknown view_type: {view_type}. Must be one of ['fused', 'graph', 'hyper', 'both']")
            
            # 
            embeddings = embeddings.detach().cpu().numpy()
            
            # 
            idx2node = self.idx2node
            embeddings_dict = {}
            
            for i, embedding in enumerate(embeddings):
                embeddings_dict[idx2node[i]] = embedding
            
            return embeddings_dict