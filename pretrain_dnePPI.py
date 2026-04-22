
#################################PPI版本###########################################
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import dhg
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
from utils.utils_graph import preprocess_nxgraph
from .le import LE
from utils.walker import RandomWalker
import random


from typing import List, Tuple, Dict, Optional,Sequence
import networkx as nx
import torch
import numpy as np
import dhg

def build_local_clique_hyperedges(
    graph,
    min_clique_size: int = 2,
    max_clique_size: int = 4,
    topk_per_node: int = 10,
    max_hyperedges: int = None,
    use_two_hop: bool = False,
):
    """
    基于局部 ego-graph 构建超边，而不是全图 maximal cliques。

    参数:
        graph: networkx graph（节点已重编号为 0..N-1 更方便）
        min_clique_size: 最小 clique 大小
        max_clique_size: 最大 clique 大小，建议 3 或 4
        topk_per_node: 每个中心节点最多保留多少个局部 clique
        max_hyperedges: 全局最多保留多少条超边，None 表示不限制
        use_two_hop: 是否扩展到 2-hop ego graph（一般先 False）

    返回:
        e_list: list[list[int]]，每个元素是一条超边（节点列表）
    """
    if min_clique_size < 2:
        raise ValueError("min_clique_size should be >= 2")
    if max_clique_size < min_clique_size:
        raise ValueError("max_clique_size should be >= min_clique_size")

    deg = dict(graph.degree())
    e_dict = {}  # key: tuple(sorted(nodes)), value: score

    for v in graph.nodes():
        # 1-hop ego graph
        ego_nodes = set([v])
        nbrs_1hop = set(graph.neighbors(v))
        ego_nodes.update(nbrs_1hop)

        # 可选 2-hop
        if use_two_hop:
            for u in list(nbrs_1hop):
                ego_nodes.update(graph.neighbors(u))

        subg = graph.subgraph(ego_nodes)

        local_candidates = []

        # enumerate_all_cliques 按 clique size 递增枚举
        for clique in nx.enumerate_all_cliques(subg):
            c_len = len(clique)

            if c_len < min_clique_size:
                continue
            if c_len > max_clique_size:
                break

            # 只保留包含中心节点 v 的局部 clique
            if v not in clique:
                continue

            clique = tuple(sorted(clique))

            # 打分：优先大 clique，其次平均全局度高
            avg_deg = sum(deg[u] for u in clique) / float(len(clique))
            score = (len(clique), avg_deg)

            local_candidates.append((score, clique))

        # 每个节点只保留 top-k
        local_candidates.sort(key=lambda x: x[0], reverse=True)
        local_candidates = local_candidates[:topk_per_node]

        for score, clique in local_candidates:
            # 全局去重；若重复出现，保留更高分
            if clique not in e_dict or score > e_dict[clique]:
                e_dict[clique] = score

    # 全局排序后再截断
    e_items = sorted(e_dict.items(), key=lambda x: x[1], reverse=True)
    if max_hyperedges is not None:
        e_items = e_items[:max_hyperedges]

    e_list = [list(c) for c, _ in e_items]

    if len(e_list) == 0:
        raise RuntimeError("No local clique hyperedges found. Please relax hyperedge construction params.")

    return e_list

class DNE_pretrain:
    def __init__(self,
                 graph,
                 feat=None,
                 hidden_dim=128,
                 num_pos_features=256,
                 local_max_clique_size=4,
                 local_topk_per_node=10,
                 local_max_hyperedges=None,
                 local_use_two_hop=False):
        
        super().__init__()
        
        self.num_pos_features = num_pos_features
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.idx2node, self.node2idx = preprocess_nxgraph(graph)
        self.num_features = feat.shape[1] if feat is not None else None
        self.original_graph = nx.relabel_nodes(graph, self.node2idx)  # 原始普通图
        self.num_nodes = self.original_graph.number_of_nodes()
        
        # Mamba 编码器
        self.mamba = Mamba(d_model=self.num_pos_features,
                           d_state=128, d_conv=2, expand=1,
                           use_fast_path=True).to(self.device)
                # ===== 新增：局部 clique 超图参数 =====
        self.local_max_clique_size = local_max_clique_size
        self.local_topk_per_node = local_topk_per_node
        self.local_max_hyperedges = local_max_hyperedges
        self.local_use_two_hop = local_use_two_hop
        # 预训练时初始化的token embedding
        self.node_token_emb = None
        self.hyper_token_emb = None
        
        # 超图相关参数
        self.H0 = None
        self.hyper_alpha = None
        self.v2e = None
        self.e2v = None
        
    def _simulate_walks(self, graph, n, l, p, q, use_rejection_sampling=True):
        """Node2Vec风格的随机游走"""
        walker = RandomWalker(graph, p=p, q=q, use_rejection_sampling=use_rejection_sampling)
        walker.preprocess_transition_probs()
        walks = walker.simulate_walks(num_walks=n, walk_length=l, workers=1, verbose=1)
        return walks
    
    # ==============================
    # 1. 超图构建与初始化
    # ==============================
    def _build_clique_hypergraph(self):
        """
        使用“局部 clique 超图”替代全图 maximal cliques。
        """
        if getattr(self, "H0", None) is not None and \
        getattr(self, "v2e", None) is not None and \
        getattr(self, "e2v", None) is not None:
            return
        
        num_nodes = self.original_graph.number_of_nodes()

        # ===== 核心修改：局部 clique，而不是全图 find_cliques =====
        e_list = build_local_clique_hyperedges(
            graph=self.original_graph,
            min_clique_size=2,
            max_clique_size=self.local_max_clique_size,
            topk_per_node=self.local_topk_per_node,
            max_hyperedges=self.local_max_hyperedges,
            use_two_hop=self.local_use_two_hop,
        )
        
        if len(e_list) == 0:
            raise RuntimeError("需要非空的局部 clique 超图")
        
        HG = dhg.Hypergraph(num_nodes, e_list)
        H0 = HG.H.to_dense().to(self.device)  # 先保持和你现有代码兼容，后面再考虑 sparse
        num_v, num_e = H0.shape
        
        if getattr(self, "H0", None) is None or getattr(self, "hyper_alpha", None) is None:
            self._init_learnable_hypergraph(H0)
        
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
        self.clique_hypergraph = HG

        print(f"[DNE_pretrain] local hypergraph built: num_nodes={num_v}, num_edges={num_e}, "
            f"max_clique_size={self.local_max_clique_size}, topk_per_node={self.local_topk_per_node}")
        
    def _init_learnable_hypergraph(self, H0: torch.Tensor):
        """初始化可学习超图参数"""
        self.H0 = H0.detach()
        num_e = self.H0.shape[1]
        self.hyper_alpha = nn.Parameter(torch.zeros(num_e, device=self.device))
    
    # ==============================
    # 2. 超图核心参数计算
    # ==============================
    # def _build_gamma(self, H: torch.Tensor):
    #     """
    #     计算γ_e(v)：节点v在超边e中的归一化权重
    #     返回: list of dict, gamma[e][v] = γ_e(v)
    #     """
    #     H = H.detach()
    #     num_v, num_e = H.shape
    #     gamma = []
        
    #     for e in range(num_e):
    #         vs = self.e2v[e]
    #         if len(vs) == 0:
    #             gamma.append({})
    #             continue
    #         vals = H[vs, e]
    #         s = vals.sum()
    #         if s.item() <= 0:
    #             prob = torch.ones_like(vals) / len(vs)
    #         else:
    #             prob = vals / s
    #         gamma.append({int(v): float(p) for v, p in zip(vs, prob)})
        
    #     return gamma
    # ==============================
    # 修改后的 _build_gamma (实现结构中心性)
    # ==============================
    def _build_gamma(self, H: torch.Tensor):
        """
        计算γ_e(v)：基于结构中心性 (Structural Centrality)
        
        方案 A 定义：
        1. 优先计算节点在超边诱导子图(Induced Subgraph)中的度数。
        2. 特殊情况处理：如果超边是 Clique（全连接），内部度数全等，
           则使用节点在原始大图(Original Graph)中的全局度数作为权重，
           确保核心节点（Hubs）能获得更高的采样概率。
        """
        num_v, num_e = H.shape
        gamma = []
        
        # 预先计算全局度数（用于Tie-breaking或Clique情况）
        global_degrees = dict(self.original_graph.degree())
        
        for e in range(num_e):
            vs = self.e2v[e] # 获取超边e包含的节点列表 [v1, v2, ...]
            if len(vs) == 0:
                gamma.append({})
                continue
            
            # --- 核心修改开始 ---
            
            # 1. 提取超边诱导的子图
            # 注意：nx.subgraph 会返回一个只包含这些节点和它们之间边的子图
            subgraph = self.original_graph.subgraph(vs)
            
            # 2. 计算子图内部度数 (Internal Degree)
            # dict result: {node_id: degree_in_subgraph}
            local_deg_dict = dict(subgraph.degree(vs))
            
            # 转换为 tensor
            local_degs = torch.tensor([local_deg_dict[v] for v in vs], dtype=torch.float32)
            
            # 3. 检查是否退化（例如 Clique 中所有节点内部度数相同）
            if local_degs.max() == local_degs.min() and len(vs) > 1:
                # 如果内部度数无区分度（如在 Clique 中），使用全局度数作为权重
                # 逻辑：即使在同一个团里，连接了外部更多边的节点（Hub）更重要
                weights = torch.tensor([global_degrees[v] for v in vs], dtype=torch.float32)
            else:
                # 否则使用内部度数
                weights = local_degs

            # 4. 归一化为概率 (Softmax 或 Linear Normalize)
            # 这里使用 Linear Normalize (L1)，符合 γ ∝ degree 的定义
            total_w = weights.sum()
            
            if total_w.item() <= 1e-6:
                # 极个别情况（如孤立点集合），退化为均匀
                prob = torch.ones_like(weights) / len(vs)
            else:
                # 添加平滑项 epsilon 防止极端分布，同时保持非均匀性
                # 审稿人提到 "gamma = H / (|e| + epsilon)" 是有问题的
                # 这里我们直接对权重归一化： p_v = w_v / sum(w)
                prob = weights / total_w

            # --- 核心修改结束 ---
            
            gamma.append({int(v): float(p) for v, p in zip(vs, prob)})
        
        return gamma
    # def _compute_edge_weights(self, H: torch.Tensor,
    #                         node_feat_for_omega: torch.Tensor,
    #                         lambda_omega: float = 1.0,
    #                         use_simplified: bool = False):
    #     """
    #     严格对齐论文的超边权重计算：
    #     - use_simplified=True: 直接用 ω(e) ∝ dens(e) / sqrt(|e|)
    #     - use_simplified=False: 用 (dens * |e|^2) / (|e|^2 + λ |e|^{β}) 一般形式
    #     """
    #     num_v, num_e = H.shape
    #     device = H.device

    #     # 没有特征时，退化为全1
    #     if node_feat_for_omega is None:
    #         omega = torch.ones(num_e, device=device)
    #         return omega.clamp(min=1e-6)

    #     z = node_feat_for_omega  # [N, D]

    #     dens_list = []
    #     size_list = []

    #     for e in range(num_e):
    #         vs = self.e2v[e]
    #         if len(vs) == 0:
    #             dens_list.append(0.0)
    #             size_list.append(1)
    #             continue

    #         z_e = z[vs]  # [|e|, D]
    #         # S_{uv} = sim(z_u, z_v)，这里用 cosine similarity 实现 sim
    #         sim = F.cosine_similarity(z_e.unsqueeze(1), z_e.unsqueeze(0), dim=-1)
    #         density = sim.mean().item()   # dens(e) = 平均相似度
    #         dens_list.append(max(density, 0.0))
    #         size_list.append(len(vs))

    #     dens = torch.tensor(dens_list, dtype=torch.float32, device=device)   # [E]
    #     size = torch.tensor(size_list, dtype=torch.float32, device=device)   # [E]

    #     if use_simplified:
    #         # 论文最终的 practical scheme:
    #         # ω(e) ∝ dens(e) / sqrt(|e|)
    #         omega = dens / size.sqrt().clamp(min=1.0)
    #     else:
    #         # 更完整的形式：ω(e) ≈ dens(e) * |e|^2 / (|e|^2 + λ |e|^{β})
    #         beta_omega = 2.5  # 论文中选择 β_ω = 5/2 对应 |e|^{-1/2}
    #         numerator = dens * (size ** 2)
    #         denominator = size ** 2 + lambda_omega * (size ** beta_omega)
    #         omega = numerator / denominator.clamp(min=1e-6)

    #     # 全局归一化（只影响一个常数因子，不改变转移分布）
    #     omega = omega / omega.mean().clamp(min=1e-6)
    #     return omega.clamp(min=1e-6)
    
    # def _compute_edge_weights(self, H: torch.Tensor, node_feat_for_omega: torch.Tensor = None):
    #     """
    #     计算优化后的超边权重ω(e)
    #     """
    #     num_v, num_e = H.shape
    #     device = H.device

    #     # 基础权重：考虑超边大小
    #     base_omega = []
    #     for e in range(num_e):
    #         vs = self.e2v[e]
    #         if len(vs) == 0:
    #             base_omega.append(1.0)
    #         else:
    #             base_omega.append(1.0 / np.sqrt(len(vs)))
    #     base_omega = torch.tensor(base_omega, dtype=torch.float32, device=device)

    #     if node_feat_for_omega is None:
    #         return base_omega.clamp(min=1e-6)

    #     # 基于节点特征计算超边内部一致性
    #     z = node_feat_for_omega  # [N, D]
    #     omega_list = []
    #     for e in range(num_e):
    #         vs = self.e2v[e]
    #         if len(vs) < 2:
    #             omega_list.append(1.0)
    #             continue
    #         z_e = z[vs]  # [|e|, D]
    #         sim = F.cosine_similarity(z_e.unsqueeze(1), z_e.unsqueeze(0), dim=-1)
    #         density = sim.mean().item()
    #         omega_list.append(max(density, 0.0))

    #     omega_struct = torch.tensor(omega_list, dtype=torch.float32, device=device)
        
    #     # 结合优化算法中的公式，调整ω(e)
    #     # ω(e) = (rho(e) / (lambda * |e|^k)) * omega_struct
    #     # 这里rho(e)通过节点相似性度量（density）来近似
    #     rho = omega_struct  # 用超边的相似性（density）作为rho(e)
    #     lambda_param = 0.2  # 可以调整的正则化参数
    #     kappa = 0.2  # 惩罚因子，控制超边大小的影响
        
    #     omega_optimized = (rho / (lambda_param * len(vs) ** kappa)) * omega_struct

    #     # 归一化和限制最小值
    #     omega_optimized = omega_optimized / omega_optimized.mean().clamp(min=1e-6)
    #     return omega_optimized.clamp(min=1e-6)
    def _compute_edge_weights(self, H: torch.Tensor, node_feat_for_omega: torch.Tensor = None):
        """
        计算优化后的超边权重 ω(e)，与论文闭式解一致：

            ω*(e) = <S, A^(e)> / (||A^(e)||_F^2 + λ_ω |e|^{β_ω})
                = (sum_{u,v in e} sim(z_u, z_v)) / (|e|^2 + λ_ω |e|^{β_ω})

        其中 A^(e)_{uv}=1 iff u,v∈e，因此 ||A^(e)||_F^2 = |e|^2。
        """
        num_v, num_e = H.shape
        device = H.device

        # 论文版本需要 S_uv=sim(z_u,z_v)，因此必须给 node_feat_for_omega
        if node_feat_for_omega is None:
            # 若你一定要保留无特征 fallback，可用纯 size penalty；否则建议直接报错
            # raise ValueError("node_feat_for_omega is required to compute ω(e) consistent with the paper.")
            omega = torch.ones(num_e, dtype=torch.float32, device=device)
            return omega.clamp(min=1e-6)

        # 读取论文超参（优先用 self 里设置的值；没有则给默认）
        lambda_omega = float(getattr(self, "lambda_omega", 0.2))
        beta_omega   = float(getattr(self, "beta_omega", 0.2))
        eps          = 1e-12

        z = node_feat_for_omega.to(device)              # [N, D]
        z = F.normalize(z, dim=-1)                      # cosine 相似度需要归一化

        omega_list = []

        for e in range(num_e):
            vs = self.e2v[e]
            m = len(vs)

            if m == 0:
                omega_list.append(torch.tensor(0.0, device=device))
                continue

            # m==1 时：sum_{u,v in e} S_uv = S_vv = 1（cosine）
            if m == 1:
                numerator = torch.tensor(1.0, device=device)
            else:
                z_e = z[vs]                             # [m, D]
                # cosine similarity matrix: sim_{ij} = <z_i, z_j>
                sim = z_e @ z_e.t()                     # [m, m]
                numerator = sim.sum()                   # sum_{u,v in e} S_uv（含对角线）

            denom = (m * m) + (lambda_omega * (m ** beta_omega))
            omega_e = numerator / (denom + eps)

            # 约束 ω(e) >= 0（论文里有非负约束；闭式解在近似下可能出现负值时可截断）
            omega_list.append(torch.clamp(omega_e, min=0.0))

        omega = torch.stack(omega_list).to(torch.float32)

        # 可选：归一化到均值为 1，便于数值稳定（不改变相对比例）
        omega = omega / omega.mean().clamp(min=1e-6)

        return omega.clamp(min=1e-6)
    
    # ==============================
    # 3. 重构后的序列采样方法
    # ==============================
    def _sample_node_walks_node2vec(self, num_walks_per_node: int, walk_len: int,
                                   p: float = 1.0, q: float = 1.0):
        """
        节点游走：在原始普通图上使用Node2Vec
        """
        # 确保图是连通的（处理孤立节点）
        graph = self.original_graph.copy()
        isolated_nodes = list(nx.isolates(graph))
        if isolated_nodes:
            for node in isolated_nodes:
                # 为孤立节点添加自环
                graph.add_edge(node, node)
        
        walks_list = self._simulate_walks(
            graph=graph,
            n=num_walks_per_node,
            l=walk_len,
            p=p,
            q=q,
            use_rejection_sampling=True
        )
        
        return walks_list
    

    def _sample_hyperedge_walks_edrw(
        self,
        num_walks_per_edge: int,
        walk_len: int,
        H: torch.Tensor,
        gamma: List[Dict],
        omega: torch.Tensor,
        edge_indices: Optional[Sequence[int]] = None,
    ) -> List[List[int]]:
        """
        超边游走：使用超图 EDRW (e→v→e'→v'...)
        返回超边 ID 序列（list[list[int]]）

        参数：
            num_walks_per_edge : 每条超边起点采样多少条路径
            walk_len           : 每条路径长度
            H                  : [num_v, num_e] 超图关联矩阵
            gamma              : γ_e(v)，由 _build_gamma 生成
            omega              : ω(e)，由 _compute_edge_weights 生成
            edge_indices       : 只在这些超边上做游走（用于大图采样），
                                若为 None，则默认使用所有超边 range(num_e)
        """
        num_v, num_e = H.shape
        device = H.device
        walks: List[List[int]] = []

        # 若未指定，则遍历所有超边
        if edge_indices is None:
            edge_indices = range(num_e)

        for e0 in edge_indices:
            # 确保是 Python int
            e0 = int(e0)

            # 起始超边为空时，直接重复自身
            if len(self.e2v[e0]) == 0:
                for _ in range(num_walks_per_edge):
                    walks.append([e0] * walk_len)
                continue

            # 从该超边 e0 开始采样多条长度为 walk_len 的路径
            for _ in range(num_walks_per_edge):
                cur_edge = e0
                path = [cur_edge]

                for _step in range(walk_len - 1):
                    # ---------- Step1: e -> v ----------
                    nodes_in_e = self.e2v[cur_edge]
                    if not nodes_in_e:
                        # 空超边，停在当前超边
                        path.append(cur_edge)
                        continue

                    # P(v | e) ∝ γ_e(v)
                    probs_v = [gamma[cur_edge].get(v, 0.0) for v in nodes_in_e]
                    total_v = sum(probs_v)

                    if total_v <= 1e-12:
                        # 退化为均匀分布
                        selected_node = random.choice(nodes_in_e)
                    else:
                        probs_v = [p / total_v for p in probs_v]
                        selected_node = random.choices(nodes_in_e, weights=probs_v, k=1)[0]

                    # ---------- Step2: v -> e' ----------
                    incident_edges = self.v2e[selected_node]
                    if not incident_edges:
                        # 该节点无关联超边，停在当前超边
                        next_edge = cur_edge
                    else:
                        e_weights = []
                        for e in incident_edges:
                            g_v = gamma[e].get(selected_node, 0.0)
                            e_weights.append(float(omega[e].item()) * g_v)

                        total_e = sum(e_weights)
                        if total_e <= 1e-12:
                            next_edge = random.choice(incident_edges)
                        else:
                            probs_e = [w / total_e for w in e_weights]
                            next_edge = random.choices(incident_edges, weights=probs_e, k=1)[0]

                    next_edge = int(next_edge)
                    path.append(next_edge)
                    cur_edge = next_edge

                walks.append(path)

        return walks    
    def _pad_sequences(self, walks_list: List[List[int]], target_len: int, 
                      padding_strategy: str = 'wrap'):
        """
        序列padding策略
        padding_strategy: 'wrap'（循环）, 'reflect'（反射）, 'repeat_last'（重复最后一个）
        """
        padded_walks = []
        
        for walk in walks_list:
            if len(walk) >= target_len:
                padded_walks.append(walk[:target_len])
            else:
                if padding_strategy == 'wrap':
                    # 循环填充
                    repetitions = (target_len + len(walk) - 1) // len(walk)
                    extended = (walk * repetitions)[:target_len]
                    padded_walks.append(extended)
                    
                elif padding_strategy == 'reflect':
                    # 反射填充
                    extended = []
                    forward = True
                    idx = 0
                    
                    while len(extended) < target_len:
                        extended.append(walk[idx])
                        if forward and idx == len(walk) - 1:
                            forward = False
                        elif not forward and idx == 0:
                            forward = True
                        idx = idx + 1 if forward else idx - 1
                    
                    padded_walks.append(extended[:target_len])
                    
                else:  # 'repeat_last' (默认)
                    padded_walks.append(walk + [walk[-1]] * (target_len - len(walk)))
        
        return padded_walks
        
    # def _refactored_sample_hypergraph_sequences(self,
    #                                             H: torch.Tensor,
    #                                             walk_len: int,
    #                                             num_walks_per_node: int,
    #                                             node_feat_for_omega: torch.Tensor = None,
    #                                             node2vec_p: float = 1.0,
    #                                             node2vec_q: float = 1.0):
    #     """
    #     重构后的序列采样接口，使用优化后的超边权重ω(e)
    #     """
    #     device = H.device
        
    #     # 确保超图结构已构建
    #     self._build_clique_hypergraph()
    #     num_v, num_e = H.shape
        
    #     # 计算超图参数（用于超边游走）
    #     gamma = self._build_gamma(H)
    #     omega = self._compute_edge_weights(H, node_feat_for_omega=node_feat_for_omega)
        
    #     # ---------- 1. 节点游走：使用普通图Node2Vec ----------
    #     node_walks_list = self._sample_node_walks_node2vec(
    #         num_walks_per_node=num_walks_per_node,
    #         walk_len=walk_len,
    #         p=node2vec_p,
    #         q=node2vec_q
    #     )
        
    #     # Padding处理
    #     padded_node_walks = self._pad_sequences(node_walks_list, walk_len, padding_strategy='wrap')
    #     node_walks = torch.tensor(padded_node_walks, device=device, dtype=torch.long)
        
    #     # 节点起始ID：每个节点重复num_walks_per_node次
    #     node_start_ids = torch.arange(num_v, device=device).repeat_interleave(num_walks_per_node)
        
    #     # ---------- 2. 超边游走：使用超图EDRW ----------
    #     hyper_walks_list = self._sample_hyperedge_walks_edrw(
    #         num_walks_per_edge=num_walks_per_node,  # 保持与节点相同的采样密度
    #         walk_len=walk_len,
    #         H=H,
    #         gamma=gamma,
    #         omega=omega  # 使用优化后的omega
    #     )
        
    #     # Padding处理
    #     padded_hyper_walks = self._pad_sequences(hyper_walks_list, walk_len, padding_strategy='wrap')
    #     hyper_walks = torch.tensor(padded_hyper_walks, device=device, dtype=torch.long)
        
    #     # 超边起始ID：每个超边重复num_walks_per_node次
    #     hyper_start_ids = torch.arange(num_e, device=device).repeat_interleave(num_walks_per_node)
        
    #     return node_walks, node_start_ids, hyper_walks, hyper_start_ids
    def _refactored_sample_hypergraph_sequences(
        self,
        H: torch.Tensor,
        walk_len: int,
        num_walks_per_node: int,
        node_feat_for_omega: torch.Tensor = None,
        node2vec_p: float = 1.0,
        node2vec_q: float = 1.0,
        max_edges_for_walks: int = 20000,   # ⭐ 新增：每个 epoch 最多多少条超边参与 Mamba
    ):
        """
        重构后的序列采样接口，使用优化后的超边权重 ω(e)，并在大图上对超边做采样。

        返回：
            node_walks      : [B_node, L]   节点随机游走序列（节点 ID）
            node_start_ids  : [B_node]      每条节点序列对应的起始节点 ID
            hyper_walks     : [B_edge, L]   超边随机游走序列（超边 ID）
            hyper_start_ids : [B_edge]      每条超边序列对应的起始超边 ID
        """
        device = H.device

        # 确保超图结构已构建 (self.e2v / self.v2e / H0 等)
        self._build_clique_hypergraph()
        num_v, num_e = H.shape

        # 计算 γ_e(v) 与 ω(e)
        gamma = self._build_gamma(H)
        omega = self._compute_edge_weights(H, node_feat_for_omega=node_feat_for_omega)

        # ---------- 1. 节点游走：使用普通图 Node2Vec ----------
        node_walks_list = self._sample_node_walks_node2vec(
            num_walks_per_node=num_walks_per_node,
            walk_len=walk_len,
            p=node2vec_p,
            q=node2vec_q,
        )
        padded_node_walks = self._pad_sequences(
            node_walks_list, walk_len, padding_strategy="wrap"
        )
        node_walks = torch.tensor(
            padded_node_walks, device=device, dtype=torch.long
        )  # [B_node, L]

        # 每个节点重复 num_walks_per_node 次
        node_start_ids = torch.arange(
            num_v, device=device, dtype=torch.long
        ).repeat_interleave(num_walks_per_node)  # [B_node]

        # ---------- 2. 超边游走：使用超图 EDRW + 超边采样 ----------
        # 对于超边数量非常大的情况，不必每个 epoch 都用全部超边
        if (
            max_edges_for_walks is not None
            and max_edges_for_walks > 0
            and num_e > max_edges_for_walks
        ):
            # 随机采样部分超边参与 EDRW
            edge_indices_np = np.random.choice(
                num_e, size=max_edges_for_walks, replace=False
            )
            edge_indices = sorted(edge_indices_np.tolist())
        else:
            # 超边数量本身不大，直接用全部
            edge_indices = list(range(num_e))

        hyper_walks_list = self._sample_hyperedge_walks_edrw(
            num_walks_per_edge=num_walks_per_node,  # 保持与节点相同的采样密度
            walk_len=walk_len,
            H=H,
            gamma=gamma,
            omega=omega,
            edge_indices=edge_indices,  # ⭐ 只在采样到的这些超边上做 EDRW
        )
        padded_hyper_walks = self._pad_sequences(
            hyper_walks_list, walk_len, padding_strategy="wrap"
        )
        hyper_walks = torch.tensor(
            padded_hyper_walks, device=device, dtype=torch.long
        )  # [B_edge, L]

        # 对应的起始超边 ID（注意：起始 ID 要用原始的超边编号，而不是 0..len(edge_indices)-1）
        # len(edge_indices) * num_walks_per_node == B_edge
        hyper_start_ids = torch.tensor(
            np.repeat(edge_indices, num_walks_per_node),
            device=device,
            dtype=torch.long,
        )  # [B_edge]

        return node_walks, node_start_ids, hyper_walks, hyper_start_ids    
    # ==============================
    # 4. 高级采样选项
    # ==============================
    def _sample_with_structural_ordering(self, H: torch.Tensor, walk_len: int,
                                        num_walks_per_node: int, 
                                        ordering_method: str = 'centrality'):
        """
        基于结构重要性排序的采样
        ordering_method: 'centrality'（中心性）, 'pagerank', 'clustering'
        """
        # 计算节点重要性
        if ordering_method == 'centrality':
            centrality = nx.degree_centrality(self.original_graph)
            node_importance = {node: centrality.get(node, 0.0) for node in range(self.num_nodes)}
        elif ordering_method == 'pagerank':
            pagerank = nx.pagerank(self.original_graph)
            node_importance = {node: pagerank.get(node, 0.0) for node in range(self.num_nodes)}
        elif ordering_method == 'clustering':
            clustering = nx.clustering(self.original_graph)
            node_importance = {node: clustering.get(node, 0.0) for node in range(self.num_nodes)}
        else:
            node_importance = {node: 1.0 for node in range(self.num_nodes)}
        
        # 按重要性排序节点
        sorted_nodes = sorted(range(self.num_nodes), 
                             key=lambda x: node_importance.get(x, 0.0), 
                             reverse=True)
        
        # 按重要性顺序采样（而不是随机顺序）
        device = H.device
        node_walks = []
        node_start_ids = []
        
        for node in sorted_nodes:
            for _ in range(num_walks_per_node):
                # 从该节点开始游走
                walk = self._simulate_walks(
                    graph=self.original_graph,
                    n=6,
                    l=6,
                    p=0.5,
                    q=1.0,
                    use_rejection_sampling=True
                )[0]
                
                padded_walk = self._pad_sequences([walk], walk_len, padding_strategy='wrap')[0]
                node_walks.append(padded_walk)
                node_start_ids.append(node)
        
        node_walks = torch.tensor(node_walks, device=device, dtype=torch.long)
        node_start_ids = torch.tensor(node_start_ids, device=device, dtype=torch.long)
        
        # 超边采样保持原样
        gamma = self._build_gamma(H)
        omega = self._compute_edge_weights(H)
        
        hyper_walks_list = self._sample_hyperedge_walks_edrw(
            num_walks_per_edge=num_walks_per_node,
            walk_len=walk_len,
            H=H,
            gamma=gamma,
            omega=omega
        )
        
        padded_hyper_walks = self._pad_sequences(hyper_walks_list, walk_len, padding_strategy='wrap')
        hyper_walks = torch.tensor(padded_hyper_walks, device=device, dtype=torch.long)
        hyper_start_ids = torch.arange(H.shape[1], device=device).repeat_interleave(num_walks_per_node)
        # 在 _refactored_sample_hypergraph_sequences 结尾处，生成 node_start_ids 时不要自己造
        node_start_ids = node_walks[:, 0].clone()
        hyper_start_ids = hyper_walks[:, 0].clone()

        return node_walks, node_start_ids, hyper_walks, hyper_start_ids
    
    # ==============================
    # 5. Mamba编码部分（保持不变）
    # ==============================
    def _encode_with_mamba(self,
                          node_walks, node_start_ids,
                          hyper_walks, hyper_start_ids,
                          H, d_model):
        """
        使用Mamba编码序列
        """
        device = self.device
        num_v, num_e = H.shape
        
        # 1) token embedding
        node_hidden = self.node_token_emb(node_walks)      # [B_node, L, D]
        hyper_hidden = self.hyper_token_emb(hyper_walks)   # [B_edge, L, D]
        
        # 2) Mamba序列编码
        node_out = self.mamba(node_hidden)    # [B_node, L, D]
        hyper_out = self.mamba(hyper_hidden)  # [B_edge, L, D]
        
        # 3) 节点序列 → 节点embedding
        node_emb_from_seq = torch.zeros(num_v, d_model, device=device)
        counts_v = torch.zeros(num_v, device=device)
        
        for i in range(node_out.size(0)):
            v = int(node_start_ids[i].item())
            seq_mean = node_out[i].mean(dim=0)
            node_emb_from_seq[v] += seq_mean
            counts_v[v] += 1
        
        node_emb_from_seq /= counts_v.unsqueeze(1).clamp(min=1.0)
        
        # 4) 超边序列 → 超边embedding
        hyper_emb_from_seq = torch.zeros(num_e, d_model, device=device)
        counts_e = torch.zeros(num_e, device=device)
        
        for i in range(hyper_out.size(0)):
            e = int(hyper_start_ids[i].item())
            seq_mean = hyper_out[i].mean(dim=0)
            hyper_emb_from_seq[e] += seq_mean
            counts_e[e] += 1
        
        hyper_emb_from_seq /= counts_e.unsqueeze(1).clamp(min=1.0)
        
        # 5) 超边embedding通过H聚合到节点
        H_row_sum = H.sum(dim=1, keepdim=True).clamp(min=1.0)
        node_emb_from_hyper = torch.mm(H, hyper_emb_from_seq) / H_row_sum  # [V, D]
        
        return node_emb_from_seq, hyper_emb_from_seq, node_emb_from_hyper
    def _symmetric_dcl(self, z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.2):
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        logits_12 = torch.matmul(z1, z2.t()) / tau   # [N, N]
        logits_21 = torch.matmul(z2, z1.t()) / tau   # [N, N]

        pos_12 = torch.diag(logits_12)               # [N]
        pos_21 = torch.diag(logits_21)               # [N]

        mask = torch.eye(logits_12.size(0), device=logits_12.device, dtype=torch.bool)

        neg_12 = logits_12.masked_fill(mask, float('-inf'))
        neg_21 = logits_21.masked_fill(mask, float('-inf'))

        loss_12 = (-pos_12 + torch.logsumexp(neg_12, dim=1)).mean()
        loss_21 = (-pos_21 + torch.logsumexp(neg_21, dim=1)).mean()

        return 0.5 * (loss_12 + loss_21)
    # ==============================
    # 6. 预训练主函数（适配重构）
    # ==============================
    def pretrain_mamba(self,
                                 num_pos_features: int = 256,
                                 walk_len: int =10,
                                 num_walks_per_node: int = 6,
                                 epochs: int = 6,
                                 neg_ratio: int = 5,
                                 lambda_align: float = 0.8,
                                 node2vec_p: float = 0.5,
                                 node2vec_q: float = 1.0,
                                 save_dir: str = "../dne_retaion"):
        """
        重构后的预训练函数
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        device = self.device
        d_model = num_pos_features
        
        # 构建超图
        self._build_clique_hypergraph()
        H = self.H0.to(device)
        num_v, num_e = H.shape
        
        # 初始化token embedding
        if self.node_token_emb is None:
            self.node_token_emb = nn.Embedding(num_v, d_model).to(device)
        if self.hyper_token_emb is None:
            self.hyper_token_emb = nn.Embedding(num_e, d_model).to(device)
        
        # 准备教师信号（谱位置编码）
        x_posLE = self.compute_pos_embLE(self.original_graph, num_pos_features=d_model)
        z_teacher = torch.from_numpy(x_posLE).float().to(device)  # [N, D]
        
        # 优化器
        optimizer = torch.optim.Adam(
            list(self.mamba.parameters()) +
            list(self.node_token_emb.parameters()) +
            list(self.hyper_token_emb.parameters()),
            lr=1e-3
        )
        
        # 预训练循环
        for epoch in range(epochs):
            # 使用重构后的采样方法
            # node_walks, node_start_ids, hyper_walks, hyper_start_ids = \
            #     self._refactored_sample_hypergraph_sequences(
            #         H=H,
            #         walk_len=walk_len,
            #         num_walks_per_node=num_walks_per_node,
            #         node_feat_for_omega=z_teacher,
            #         node2vec_p=node2vec_p,
            #         node2vec_q=node2vec_q
            #     )
            node_walks, node_start_ids, hyper_walks, hyper_start_ids = \
                self._refactored_sample_hypergraph_sequences(
                    H=H,
                    walk_len=walk_len,
                    num_walks_per_node=num_walks_per_node,
                    node_feat_for_omega=z_teacher,
                    node2vec_p=node2vec_p,
                    node2vec_q=node2vec_q,
                    max_edges_for_walks=None,  # 可按显存调，比如 10000 / 20000 / 50000
                )
                        
            # Mamba编码
            node_emb_from_seq, hyper_emb_from_seq, node_emb_from_hyper = \
                self._encode_with_mamba(node_walks, node_start_ids,
                                       hyper_walks, hyper_start_ids,
                                       H, d_model)
            
            # 最终节点embedding（可以选择不同组合）
            # final_node_emb = node_emb_from_seq  # 仅节点序列
            # final_node_emb = node_emb_from_hyper  # 仅超边聚合
            final_node_emb =  node_emb_from_hyper # 两者结合
            
            # 构造对比学习样本
            pos_pairs = self._extract_pos_pairs_from_walks(node_walks, node_start_ids)
            neg_pairs = self._sample_negatives_for_pairs(pos_pairs, num_nodes=num_v,
                                                        neg_ratio=neg_ratio)
            
            if pos_pairs.size(0) == 0 or neg_pairs.size(0) == 0:
                print(f"[Mamba pretrain] epoch {epoch+1}: 没有正负样本对，跳过")
                continue
            
            # 对比损失
            i_pos = pos_pairs[:, 0]
            j_pos = pos_pairs[:, 1]
            i_neg = neg_pairs[:, 0]
            j_neg = neg_pairs[:, 1]
            
            z = final_node_emb
            tau = 0.2  # 甚至 0.2，都可以试
            pos_score = (z[i_pos] * z[j_pos]).sum(-1) / tau
            neg_score = (z[i_neg] * z[j_neg]).sum(-1) / tau
            
            loss_pos = F.softplus(-pos_score).mean()
            loss_neg = F.softplus(neg_score).mean()
            # loss_contrast = loss_pos + loss_neg
            z_seq = node_emb_from_seq
            z_hyper = node_emb_from_hyper
            loss_contrast = self._symmetric_dcl(z_seq, z_hyper, tau=tau)
            # 对齐损失
            loss_align = F.mse_loss(final_node_emb, z_teacher)
            
            # 总损失
            loss = loss_contrast + lambda_align * loss_align
            
            # 优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"[Mamba pretrain] epoch {epoch+1}/{epochs}, "
                  f"loss={loss.item():.4f}, "
                  f"contrast={loss_contrast.item():.4f}, "
                  f"align={loss_align.item():.4f}")
        
        # 保存模型和embedding
        self._save_pretrained_results(H, z_teacher, d_model, save_dir)
        
        print(f"[Mamba pretrain] 完成。结果保存至: {save_dir}")
    
    def _save_pretrained_results(self, H, z_teacher, d_model, save_dir):
        """保存预训练结果"""
        import os
        import torch
        import numpy as np
        
        with torch.no_grad():
            # node_walks, node_start_ids, hyper_walks, hyper_start_ids = \
            #     self._refactored_sample_hypergraph_sequences(
            #         H=H,
            #         walk_len=3,
            #         num_walks_per_node=3,
            #         node_feat_for_omega=z_teacher
            #     )
            node_walks, node_start_ids, hyper_walks, hyper_start_ids = \
                self._refactored_sample_hypergraph_sequences(
                    H=H,
                    walk_len=6,
                    num_walks_per_node=3,
                    node_feat_for_omega=z_teacher
                )
            
            node_emb_from_seq, hyper_emb_from_seq, node_emb_from_hyper = \
                self._encode_with_mamba(node_walks, node_start_ids,
                                       hyper_walks, hyper_start_ids,
                                       H, d_model)
            
            final_node_emb = node_emb_from_hyper
            x_pos_node = node_emb_from_seq.detach().cpu().numpy()
            x_pos_hyper = final_node_emb.detach().cpu().numpy()
        
        # 保存embedding
        np.save(os.path.join(save_dir, "x_posMamb_atha.npy"), x_pos_node)
        np.save(os.path.join(save_dir, "x_posHMamb_atha.npy"), x_pos_hyper)
        
        # 保存模型
        torch.save(
            {
                "mamba_state_dict": self.mamba.state_dict(),
                "node_token_emb": self.node_token_emb.state_dict(),
                "hyper_token_emb": self.hyper_token_emb.state_dict(),
                "num_pos_features": d_model,
                "num_nodes": H.shape[0],
                "num_edges": H.shape[1],
            },
            os.path.join(save_dir, "mamba_hyper_pretrain_refactored.pt"),
        )
        
        self.x_posMamba = x_pos_node
        self.x_posHMamba = x_pos_hyper
    
    # ==============================
    # 辅助函数（保持不变）
    # ==============================
    def _extract_pos_pairs_from_walks(self, node_walks, node_start_ids):
        """从节点序列中提取正样本对"""
        device = node_walks.device
        B, L = node_walks.shape
        pairs = []
        
        for idx in range(B):
            i = int(node_start_ids[idx].item())
            seq = node_walks[idx]
            for t in range(1, L):
                j = int(seq[t].item())
                if i == j:
                    continue
                pairs.append([i, j])
        
        if len(pairs) == 0:
            return torch.empty((0, 2), dtype=torch.long, device=device)
        return torch.tensor(pairs, dtype=torch.long, device=device)
    
    def _sample_negatives_for_pairs(self, pos_pairs, num_nodes, neg_ratio: int):
        """负采样"""
        device = pos_pairs.device
        P = pos_pairs.size(0)
        if P == 0 or neg_ratio <= 0:
            return torch.empty((0, 2), dtype=torch.long, device=device)
        
        neg_list = []
        for idx in range(P):
            i = int(pos_pairs[idx, 0].item())
            for _ in range(neg_ratio):
                k = np.random.randint(0, num_nodes)
                if k == i:
                    k = (k + 1) % num_nodes  # 避免自环
                neg_list.append([i, k])
        
        if len(neg_list) == 0:
            return torch.empty((0, 2), dtype=torch.long, device=device)
        return torch.tensor(neg_list, dtype=torch.long, device=device)
    
    def compute_pos_embLE(self, graph, num_pos_features=256):
        """计算谱位置编码（示例函数）"""
        # 这里需要实现具体的LE计算
        # 返回: [N, num_pos_features] 的numpy数组
        pos_emb_model = LE(graph, num_pos_features)
        return pos_emb_model._X
