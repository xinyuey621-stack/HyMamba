import sys
import os

import torch.multiprocessing as mp

# from models.GIN_DNE_Res_KL_qianghua_speedPPO import FreeKD_DNE_QL_speedPPO
from models.dne import DNE
from models.le import LE
from models.lle import LLE
import torch
from models.pretrain_dne import DNE_pretrain
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from models.deepwalk import DeepWalk


class NodeEmbedding_speed:
    def __init__(self, args=None, graph=None, graph_test = None,node_label=None,neighbors_dict=None, name='Train_graph'):
        self.args = args
        self.graph = graph
        self.name = name
        self.node_label = node_label
        self.graph_test = graph_test
        self.neighbors_dict_all = neighbors_dict
        self._spawn_method_set = False  # 

    def get_embeddings(self, method, feat=None,node_label=None, feat_text=None, embed_size=128):
        print(f'Generate {method} embeddings for {self.name}')
        # Initialize embeddings to None or default values
        embeddings = None
        embedding1 = None
        embedding2 = None

        if method == 'DNE':
            batch_size = self.args.batch_size if self.args else 5120
            epochs = self.args.epochs if self.args else 10
            walk_number = self.args.walk_number if self.args else 100
            walk_length = self.args.walk_length if self.args else 10
            p = self.args.p if self.args else 1.0
            q = self.args.q if self.args else 1.0
            # if feat is None:
            #     model0 = DNE_pretrain(self.graph, hidden_dim=embed_size)
            # else:
            #     model0 = DNE_pretrain(self.graph, feat=feat, hidden_dim=embed_size)
                    
            # model0.pretrain_mamba()
            
            if feat is None:
                model = DNE(self.graph, hidden_dim=embed_size)
            else:
                model = DNE(self.graph, feat=feat, hidden_dim=embed_size)
                    
            model.train(batch_size=batch_size,
                        epochs=epochs,
                        walk_number=walk_number,
                        walk_length=walk_length,
                        p=p,
                        q=q)

            embeddings,embeddings1,embeddings2 = model.get_embeddings()

        elif method == 'LLE':
            model = LLE(self.graph, embed_size)
            embeddings = model.get_embeddings()

        self.embeddings = embeddings
        self.embeddings1 = embeddings
        self.embeddings2 = embeddings

        return self.embeddings, self.embeddings1, self.embeddings2

    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        node_num = len(self.embeddings.keys())
        fout.write("{} {}\n".format(node_num, self.embed_size))
        for node, vec in self.embeddings.items():
            fout.write("{} {}\n".format(node, ' '.join([str(x) for x in vec])))
        fout.close()