
from embedding import NodeEmbedding
import os
import torch
device = torch.device('cuda:0')
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, balanced_accuracy_score, average_precision_score
from sklearn import model_selection
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append("..")
from utils.edge_splitter import EdgeSplitter
from embedding_speed import NodeEmbedding_speed
from embedding import NodeEmbedding

import numpy
import random
from xgboost import XGBClassifier

def split_train_test_edges(graph, test_size, seed):

    edge_splitter_test = EdgeSplitter(graph)
    graph_test, X_test, Y_test = edge_splitter_test.train_test_split(
        p=0.1,
        keep_connected=False,
        seed=seed
    )

    edge_splitter_train = EdgeSplitter(graph_test)
    graph_train, X, Y = edge_splitter_train.train_test_split(
        p=0.1,
        keep_connected=False,
        seed=seed
    )
    (
        X_train,
        X_valid,
        Y_train,
        Y_valid,
    ) = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)

    # print('len(labels_train): ', len(Y_train))
    # print('len(labels_test): ', len(Y_test))
    # print('len(labels_valid):', len(Y_valid))

    examples = (X_train, X_test, X_valid)
    labels = (Y_train, Y_test, Y_valid)
    # 
    return graph_train, graph_test,examples, labels

# ----------------- Link Prediction -----------------
def operator_hadamard(u, v):
    return u * v
def operator_l1(u, v):#
    return np.abs(u - v)
def operator_l2(u, v):#
    return (u - v) ** 2
def operator_avg(u, v):#
    return (u + v) / 2.0

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def precompute_neighbors(G):

    neighbors_dict = {}
    for node in G.nodes():
        neighbors_dict[node] = list(G.neighbors(node))
    return neighbors_dict
binary_operators = [operator_hadamard, operator_l1, operator_l2, operator_avg]
class LinkPredictor(object):
    def __init__(self, args=None, graph=None, results_path=None):
        self.args = args  # 
        self.graph = graph  # 
        self.results_path = results_path  # 
        
        self.embed_size = self.args.embed_size if self.args else 128  #  128

    def train_and_evaluate(self, method, node_subjects=None, node_label = None,node_text=None, cv_fold=5, n_trials=5):
        all_score = []
        for trial in range(n_trials):
            seed = trial+1
            set_seed(seed)  # 
            graph_train, graph_test, examples, labels = split_train_test_edges(self.graph, test_size=0.3, seed=seed)
            X_train, X_test, X_valid = examples
            Y_train, Y_test, Y_valid = labels
            if method == 'FreeKD_DNE_QL_speed':
                neighbors_dict = precompute_neighbors(graph_train)
                
                embedding_model = NodeEmbedding_speed(self.args, graph_train, node_label,neighbors_dict,"Train Graph")
            
            elif method == 'FreeKD_DNE_QL_speedPPO':
                neighbors_dict = precompute_neighbors(graph_train)
                
                embedding_model = NodeEmbedding_speed(self.args, graph_train, neighbors_dict,"Train Graph")
            else:
                embedding_model = NodeEmbedding(self.args, graph_train,graph_test,"Train Graph")
            # 
            embeddings = None
            embeddings1 = None
            embeddings2 = None
            if node_subjects is not None:
                feat = node_subjects[[col for col in node_subjects.columns if col != 'node_label']].to_numpy()  # 
                embeddings,embedding1,embedding2 = embedding_model.get_embeddings(method, feat=feat,
                                                                                      embed_size=self.embed_size)  # 
                # print('1')
            else:
                embeddings,embedding1,embedding2= embedding_model.get_embeddings(method,
                                                                                      embed_size=self.embed_size)
            # 
            self.embeddings = embeddings  # 
            self.embeddings1 = embedding1 # phi 
            self.embeddings2 = embedding2  # psi 
            # 
            results_avg = self._train_and_evaluate_single_embedding(X_train, Y_train, X_valid, Y_valid, X_test, Y_test,
                                                                    self.embeddings, "Average Embedding", trial)
            results_phi = self._train_and_evaluate_single_embedding(X_train, Y_train, X_valid, Y_valid, X_test, Y_test,
                                                                    self.embeddings1, "Phi Embedding", trial)
            results_psi = self._train_and_evaluate_single_embedding(X_train, Y_train, X_valid, Y_valid, X_test, Y_test,
                                                                    self.embeddings2, "Psi Embedding", trial)

            all_score.extend([results_avg, results_phi, results_psi])

            # 
            print(f"Trial: {trial}")
            print(
                f"Average Embedding - AUC-ROC: {results_avg['score']['auc_roc']}, AUC-PR: {results_avg['score']['auc_pr']}, "
                f"Accuracy: {results_avg['score']['acc']}, F1 Score: {results_avg['score']['f1']}, "
                f"Balanced Accuracy: {results_avg['score']['bcc']}")
            print(
                f"Phi Embedding - AUC-ROC: {results_phi['score']['auc_roc']}, AUC-PR: {results_phi['score']['auc_pr']}, "
                f"Accuracy: {results_phi['score']['acc']}, F1 Score: {results_phi['score']['f1']}, "
                f"Balanced Accuracy: {results_phi['score']['bcc']}")
            print(
                f"Psi Embedding - AUC-ROC: {results_psi['score']['auc_roc']}, AUC-PR: {results_psi['score']['auc_pr']}, "
                f"Accuracy: {results_psi['score']['acc']}, F1 Score: {results_psi['score']['f1']}, "
                f"Balanced Accuracy: {results_psi['score']['bcc']}")

            # 
            if self.results_path:
                preds_path = f'{self.results_path}/{method}'
                if not os.path.exists(preds_path):
                    os.makedirs(preds_path)
                np.savez(f"{preds_path}/trial_{trial}_avg_results.npz", Y_true=Y_test, Y_pred=results_avg['Y_pred'],
                         Y_prob=results_avg['Y_prob'])
                np.savez(f"{preds_path}/trial_{trial}_phi_results.npz", Y_true=Y_test, Y_pred=results_phi['Y_pred'],
                         Y_prob=results_phi['Y_prob'])
                np.savez(f"{preds_path}/trial_{trial}_psi_results.npz", Y_true=Y_test, Y_pred=results_psi['Y_pred'],
                         Y_prob=results_psi['Y_prob'])

        return pd.DataFrame(all_score)  #  Pandas DataFrame

    def _train_and_evaluate_single_embedding(self, X_train, Y_train, X_valid, Y_valid, X_test, Y_test, embeddings, embedding_name, trial):
        results = []
        for op in binary_operators:  # 
            model = self.train(X_train, Y_train, op, embeddings)  # 
            valid_score = self.predict(model, X_valid, Y_valid, op, embeddings)  # 
            results.append(valid_score)

        best_result = max(results, key=lambda result: result["score"]["auc_roc"])
        cv_score = self.predict(best_result['classifier'], X_test, Y_test, best_result['binary_operator'], embeddings)  # 

        score = cv_score['score']
        score['method'] = embedding_name  # 
        score['trial'] = trial  # 

        return {
            "classifier": best_result['classifier'],
            "binary_operator": best_result['binary_operator'],
            "score": score,
            'Y_pred': cv_score['Y_pred'],
            'Y_prob': cv_score['Y_prob']
        }

    def train(self, X, Y, op, embeddings):

        xgb_clf = XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='auc',
            use_label_encoder=False
        )

        # 
        X_embed = op(np.array([embeddings[i] for i in np.transpose(X)[0]]),
                     np.array([embeddings[j] for j in np.transpose(X)[1]]))
        # 
        Y = Y.astype(int)
        class_weights = len(Y) / (2 * np.bincount(Y))

        clf = Pipeline([
            ("sc", StandardScaler()),
            ("clf", xgb_clf.set_params(scale_pos_weight=class_weights[1]))
        ])
        return clf.fit(X_embed, Y)
    def predict(self, clf, X, Y, op, embeddings):
        # 
        X_embed = op(np.array([embeddings[i] for i in np.transpose(X)[0]]),
                     np.array([embeddings[j] for j in np.transpose(X)[1]]))
        # 
        Y_prob = clf.predict_proba(X_embed)[:, 1]
        Y_pred = clf.predict(X_embed)
        # 
        score = {
            'auc_roc': round(roc_auc_score(Y, Y_prob), 4),
            'auc_pr': round(average_precision_score(Y, Y_prob), 4),
            'acc': round(accuracy_score(Y, Y_pred), 4),
            'f1': round(f1_score(Y, Y_pred), 4),
            'bcc': round(balanced_accuracy_score(Y, Y_pred), 4)
        }

        return {
            "classifier": clf,
            "binary_operator": op,
            "score": score,
            'Y_pred': Y_pred,
            'Y_prob': Y_prob
        }
