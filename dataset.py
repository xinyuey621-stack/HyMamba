import os

import numpy as np
import pandas as pd
import networkx as nx
from torch_geometric.datasets import Planetoid, PPI
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import AttributedGraphDataset
from torch_geometric.data import Data
from torch_geometric.datasets import Amazon
import torch_geometric.transforms as T
import itertools
from collections import defaultdict
import scipy.sparse as sp
import torch
class GraphDataset:
    def __init__(self, base_path):
        self.base_path = base_path
        self.graph = None
        self.node_subjects = None

    def load_graph(self, dataset, add_feats=False):
        if dataset in ['a_thaliana', 'c_elegans', 'ctd','HuRI',
                       'chemical_chemicallinks','chemicals_diseasesCTD','CTD_genes_diseases']:
            self.edge_fname = os.path.join(self.base_path, f"{dataset}/edge_list.csv")
            graph = read_graph(
                edge_fname=self.edge_fname,
                edge_list_separator=",",
                edge_list_header=True,
                edge_src_col=0,
                edge_dst_col=1,
                edge_list_numeric_node_ids=False,
                edge_weight_col=None,
                directed=False,
                name=dataset,
                )
            node_subjects = pd.DataFrame()
        elif dataset == 'cora':
            dataset = Planetoid(root=self.base_path, name=dataset)

            data = dataset[0]
            # print(type(data))
            if isinstance(data, dict):
                data = Data(x=data['x'], edge_index=data['edge_index'], y=data['y'])
            graph = to_networkx(data, to_undirected=True)
            if add_feats:
                node_subjects = pd.DataFrame(data.x, index=list(graph.nodes()))
            else:
                node_subjects = pd.DataFrame()
                node_labels = pd.DataFrame(data.y, index=list(graph.nodes()))
        elif dataset == 's_cerevisiae':
            self.edge_fname = os.path.join(self.base_path, f"{dataset}/edge_list.txt")
            graph = nx.read_weighted_edgelist(self.edge_fname, delimiter=' ')
            if add_feats:
                node_subjects = pd.read_csv(f'{self.base_path}/{dataset}/Krogan-2006_esm_emb_t36.csv', index_col=0)
                node_subjects.set_index("node_id", inplace=True)
                node_subjects = node_subjects.loc[list(graph.nodes()),:]
            else:
                node_subjects = pd.DataFrame()

        elif dataset in ['USAir', 'NS', 'PB', 'Power', 'Router']:
            data_dir = os.path.join(self.base_path, 'others', f'{dataset}.mat')
            import scipy.io as sio
            net = sio.loadmat(data_dir)
            graph = nx.from_scipy_sparse_array(net['net'])
            node_subjects = pd.DataFrame()
        elif dataset in ['cora_coauth', 'dblp_coauth','aminer','house','dblp_copub','imdb','news','modelnet_40']:
            # 1) load hypergraph-format data (HyperBoy-style)
            X = torch.load(os.path.join(self.base_path, f"{dataset}/X.pt"), map_location="cpu")
            H = torch.load(os.path.join(self.base_path, f"{dataset}/H.pt"), map_location="cpu")
            Y = torch.load(os.path.join(self.base_path, f"{dataset}/Y.pt"), map_location="cpu")

            num_nodes = int(X.shape[0])

            # 2) hypergraph -> pairwise graph (edge list)
            # For DBLP, topk pruning is strongly recommended to keep graph sparse.
            topk = 3 if dataset in ['dblp_coauth','aminer','house','modelnet_40'] else None  # you can tune (e.g., 20/50/100)
            edge_index, edge_weight = hypergraph_H_to_graph_edges(
                H, num_nodes=num_nodes,
                method="size_norm_clique",
                topk=topk,
                undirected=True
            )

            # 3) build NetworkX graph for your existing pipeline
            graph = nx.Graph()
            graph.add_nodes_from(range(num_nodes))

            # add weighted edges
            ei = edge_index.t().numpy()
            ew = edge_weight.numpy()
            for (u, v), w in zip(ei, ew):
                if u == v:
                    continue
                # accumulate weights if duplicated
                if graph.has_edge(int(u), int(v)):
                    graph[int(u)][int(v)]['weight'] += float(w)
                else:
                    graph.add_edge(int(u), int(v), weight=float(w))

            # 4) node features / labels
            if add_feats:
                feats = X.numpy()
                node_subjects = pd.DataFrame(
                    feats,
                    index=list(graph.nodes()),
                    columns=[f'feat_{i}' for i in range(feats.shape[1])]
                )
                node_subjects.insert(0, 'node_name', list(graph.nodes()))
                #  
                node_subjects['node_label'] = Y.numpy().astype(int)
            else:
                node_subjects = pd.DataFrame({'node_name': list(graph.nodes())})

            #  label
            node_labels = pd.DataFrame(Y.numpy().astype(int), index=list(graph.nodes()), columns=['label'])

        elif dataset in ['chameleon','facebook','Twitch_RU','texas','film','wisconsin']:
            self.edge_fname = os.path.join(self.base_path, f"{dataset}/edge_list.csv")
            graph = read_graph(
                edge_fname=self.edge_fname,
                edge_list_separator=",",
                edge_list_header=True,
                edge_src_col=0,
                edge_dst_col=1,
                edge_list_numeric_node_ids=False,
                edge_weight_col=None,
                directed=False,
                name=dataset,
                )
            node_subjects = pd.DataFrame()
            # node_labels = pd.DataFrame(data.y, index=list(graph.nodes()))
            label_path = os.path.join(self.base_path, f"{dataset}/labels.csv")
            node_labels = pd.read_csv(label_path)  #  header=None
            node_labels = pd.DataFrame(node_labels)

        elif dataset == 'PPI':
            dataset = PPI(root='./data/ppi')  #
            data = dataset[0]
            graph = to_networkx(data, to_undirected=True)

            y_onehot = data.y.numpy()  # 
            PPI_label = np.argmax(y_onehot, axis=1)  #  PPI_label
            # 
            if add_feats:
                node_features = data.x.numpy()  #
                node_subjects = pd.DataFrame(
                    data=node_features,
                    index=list(graph.nodes()),  #
                    columns=[f'feat_{i}' for i in range(node_features.shape[1])]  # 
                )
                node_subjects['node_label'] = PPI_label  # 
            else:
                node_subjects = pd.DataFrame()

        self.graph = graph
        self.node_subjects = node_subjects
        # self.node_labels = PPI_label
        self.node_labels = node_labels
    def normalize(self):
        weights = nx.get_edge_attributes(self.graph, 'weight')
        min_weight = min(weights.values())
        max_weight = max(weights.values())

        for edge, weight in weights.items():
            normalized_weight = (weight - min_weight) / (max_weight - min_weight)
            self.graph[edge[0]][edge[1]]['weight'] = normalized_weight

    def filter_edges_by_weight(self, min_edge_weight):
        edges_to_remove = [(u, v) for u, v, weight in self.graph.edges(data='weight') if weight < min_edge_weight]
        self.graph.remove_edges_from(edges_to_remove)
        self.update_node_subjects()

    def remove_self_loops(self):
        self_loop_nodes = [node for node in self.graph.nodes() if self.graph.has_edge(node, node)]
        self.graph.remove_nodes_from(self_loop_nodes)
        self.update_node_subjects()

    def remove_disconnected_nodes(self):
        isolated_nodes = list(nx.isolates(self.graph))
        self.graph.remove_nodes_from(isolated_nodes)
        self.update_node_subjects()

    def keep_only_largest_connected_components(self):
        largest_component = max(nx.connected_components(self.graph), key=len)
        nodes_to_remove = [node for node in self.graph.nodes() if node not in largest_component]
        self.graph.remove_nodes_from(nodes_to_remove)
        self.update_node_subjects()

    def update_node_subjects(self):
        graph_nodes = set(self.graph.nodes())
        self.node_subjects = self.node_subjects[self.node_subjects.node_name.isin(graph_nodes)]

def hypergraph_H_to_graph_edges(H, num_nodes, method="size_norm_clique", topk=None, undirected=True):
    """
    Convert hypergraph incidence H to pairwise graph edges.

    Parameters
    ----------
    H : torch.Tensor
        Either:
        (a) COO index: shape [2, nnz], H[0]=node_id, H[1]=hyperedge_id
        (b) Dense incidence: shape [N, M], nonzero indicates membership
    num_nodes : int
        Number of nodes N.
    method : str
        - "clique": unnormalized clique expansion
        - "size_norm_clique": clique expansion with hyperedge-size normalization (recommended)
    topk : int or None
        Keep only top-k weighted neighbors per node (recommended for large datasets like DBLP).
    undirected : bool
        Whether to output undirected edges (i.e., add both (u,v) and (v,u)).

    Returns
    -------
    edge_index : torch.LongTensor [2, E]
    edge_weight : torch.FloatTensor [E]
    """
    # ---- Step 1: build incidence lists (hyperedge -> list of nodes) ----
    if H.dim() == 2 and H.shape[0] == 2:
        # COO index format
        node_ids = H[0].detach().cpu().numpy()
        edge_ids = H[1].detach().cpu().numpy()
        num_edges = int(edge_ids.max() + 1) if edge_ids.size > 0 else 0

        buckets = [[] for _ in range(num_edges)]
        for v, e in zip(node_ids, edge_ids):
            buckets[int(e)].append(int(v))

    elif H.dim() == 2:
        # Dense incidence [N, M]
        H_cpu = H.detach().cpu()
        num_edges = H_cpu.shape[1]
        nz = (H_cpu != 0).nonzero(as_tuple=False)  # [nnz, 2] with (node, edge)
        buckets = [[] for _ in range(num_edges)]
        for v, e in nz.tolist():
            buckets[e].append(v)
    else:
        raise ValueError("Unsupported H format. Expect COO [2, nnz] or dense [N, M].")

    # ---- Step 2: clique expansion with weights ----
    # Use a sparse accumulation to avoid quadratic python dict explosion as much as possible.
    # Build COO for adjacency by enumerating per hyperedge (still O(sum |e|^2); use topk for DBLP).
    rows, cols, vals = [], [], []

    for nodes in buckets:
        m = len(nodes)
        if m <= 1:
            continue

        # hyperedge-size normalization to reduce dominance of large hyperedges
        if method == "clique":
            w = 1.0
        elif method == "size_norm_clique":
            # common choice: 1/(m-1) (you may also use 1/comb(m,2))
            w = 1.0 / float(m - 1)
        else:
            raise ValueError("method must be 'clique' or 'size_norm_clique'")

        # generate pairs
        # NOTE: for very large hyperedges, this is expensive; consider skipping or sampling if needed.
        for u, v in itertools.combinations(nodes, 2):
            rows.append(u); cols.append(v); vals.append(w)
            if undirected:
                rows.append(v); cols.append(u); vals.append(w)

    if len(rows) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_weight = torch.empty((0,), dtype=torch.float)
        return edge_index, edge_weight

    # ---- Step 3: sum duplicate edges via scipy sparse ----
    A = sp.coo_matrix((vals, (rows, cols)), shape=(num_nodes, num_nodes)).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()

    # ---- Step 4: optional top-k pruning per node (recommended for DBLP) ----
    if topk is not None and topk > 0:
        A = A.tolil()
        for i in range(num_nodes):
            row = A.data[i]
            col = A.rows[i]
            if len(row) > topk:
                # keep topk largest weights
                idx = sorted(range(len(row)), key=lambda t: row[t], reverse=True)[:topk]
                A.data[i] = [row[t] for t in idx]
                A.rows[i] = [col[t] for t in idx]
        A = A.tocsr()
        A.eliminate_zeros()

    A = A.tocoo()
    edge_index = torch.tensor([A.row, A.col], dtype=torch.long)
    edge_weight = torch.tensor(A.data, dtype=torch.float)
    return edge_index, edge_weight


def read_graph(edge_fname,
               edge_list_separator='\t',
               edge_list_header=False,
               edge_src_col=0,
               edge_dst_col=1,
               edge_list_numeric_node_ids=True,
               edge_weight_col=None,
               edge_type_col=None,
               edge_type_name=None,
               directed=False,
               name=None):
    '''
    Reads the input network in networkx.

    Parameters:
    - edge_fname: Path to the file containing the edge list.
    - edge_list_separator: Separator used in the edge file.
    - edge_list_header: Whether the edge file has a header.
    - edge_src_col: Index of the source column in the edge file.
    - edge_dst_col: Index of the destination column in the edge file.
    - edge_list_numeric_node_ids: Both src and dst columns use numeric node_ids instead of node names
    - edge_weight_col: Index of the weight column in the edge file.
    - directed: Whether the graph is directed.
    - name: Name to assign to the graph.

    Returns:
    - G: NetworkX graph.
    - node_df: Pandas DataFrame containing node information.
    '''
    G = nx.DiGraph() if directed else nx.Graph()
    with open(edge_fname, 'r') as edge_file:
        if edge_list_header:
            next(edge_file) # Skip the header

        for line in edge_file:
            edge = line.strip().split(edge_list_separator)
            src, dst = edge[edge_src_col], edge[edge_dst_col]

            # Check if weights are provided
            if edge_weight_col is not None:
                weight = float(edge[edge_weight_col])
            else:
                weight = None

            if edge_type_col is not None:
                edge_type = edge[edge_type_col]
            else:
                edge_type = None

            if edge_type is None or edge_type == edge_type_name:
                if src is not None and dst is not None:
                    if edge_list_numeric_node_ids:
                        src, dst = int(src), int(dst)
                    if weight is not None:
                        G.add_edge(src, dst, weight=weight)
                    else:
                        G.add_edge(src, dst)
    G.name = name

    return G
