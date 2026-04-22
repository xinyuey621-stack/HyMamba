import os

import random

import numpy as np
import pandas as pd
from argparse import ArgumentParser
from link_prediction import LinkPredictor
from dataset import GraphDataset
# from community import community_louvain

import torch
# Define metrics
LINK_PREDICTION_METRICS = ["auc_roc", "auc_pr", "f1", "acc", "bcc"]
MODULE_DETECTION_METRICS = ["ami"]


def parse_args():
    parser = ArgumentParser()  #
    parser.add_argument('--dataset', default='a_thaliana', type=str, choices=[
        'a_thaliana', 'ctd','c_elegans', 'HuRI', 's_cerevisiae', 'cora', 'Power', 'Router'
    ], help='dataset name')  #  --'a_thaliana'
    parser.add_argument('--task', default='link_prediction', type=str, choices=[
        'link_prediction', 'module_detection','link_prediction_heuristic'
    ], help='task to perform')  # #  --'link_prediction'
    parser.add_argument('--task_label', default='IntAct', type=str, choices=[
        'GOBP', 'IntAct', 'KEGG'
    ], help='labels for module identification')  #  --
    parser.add_argument('--add_feats', action='store_true', default=True,
                        help='use node features')  #  --
    parser.add_argument('--n_trials', default=3, type=int,
                        help='number of trials')  #  --

    # model related
    parser.add_argument('--epochs', default=30, type=int, help='training epochs')  
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate') 
    parser.add_argument('--batch_size', default=5012, type=int,
                        help='batch size')  
    parser.add_argument('--dropout', default=0.1, type=float,
                        help='dropout rate')  
    parser.add_argument('--embed_size', default=256, type=int,
                        help='embedding size')  

    # random walk related
    parser.add_argument('--walk_number', default=30, type=int,
                        help='number of random walks per node') 
    parser.add_argument('--walk_length', default=10, type=int,
                        help='length of each random walk') 
    parser.add_argument('--p', default=0.5, type=float,
                        help='p controls how fast the walk explores')  
    parser.add_argument('--q', default=1
                        , type=float,
                        help='q controls how fast the walk leaves the neighborhood of starting node')  

    # others
    parser.add_argument('--data_path', default='../data',
                        help='path to data') 
    parser.add_argument('--save_path', default='../result', help='path to save results')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    args = parser.parse_args()

    return args

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False     
    os.environ['PYTHONHASHSEED'] = str(seed)  



def main(args):
    # Load dataset
    # set_seed(args.seed)
    graph_data = GraphDataset(args.data_path)  
    graph_data.load_graph(args.dataset, add_feats=args.add_feats) 

    graph, node_subjects = graph_data.graph, graph_data.node_subjects  

    print(graph_data.node_subjects.head())

    node_subjects = graph_data.node_subjects

    if node_subjects.empty:  
        node_subjects = None  

    num_edges = graph.number_of_edges()  
    num_nodes = graph.number_of_nodes()  # 
    edge_density = num_edges / (num_nodes * (num_nodes - 1) / 2)  # 

    # Calculate average degree
    degree_sequence = list(dict(graph.degree()).values())  # 
    average_degree = sum(degree_sequence) / num_nodes  # 

    # Print graph statistics
    print("\nGraph Loaded:")      
    print(f"Number of nodes: {num_nodes}")  # 
    print(f"Number of edges: {num_edges}")  # 
    print(f"Edge density: {edge_density}")  # 
    print(f"Average degree: {average_degree}")  # 

    if args.task == 'link_prediction_heuristic':  
        methods = ['JC', 'CN', 'PA', 'RA', 'RP', 'Katz']  # 
    else:
        methods = ['DNE']  # 
        # methods = ['DNE',FreeKD_DNE,'DNE_with_GIN', GIN_DNE_Res 'DNE_with_GIN_negivate','GraRep', 'HOPE', 'NetMF', 'LLE', 'N2V', 'SVD']
    # 
    results_path = None
    if args.task == 'link_prediction':
        results_path = f'{args.save_path}/{args.task}/{args.dataset}/'
    elif args.task == 'link_prediction_heuristic':
        results_path = f'{args.save_path}/{args.task}/{args.dataset}/'
    elif args.task == 'module_detection':
        results_path = f'{args.save_path}/{args.task}/{args.dataset}/{args.task_label}'

    if results_path:
        eval_result_file = f'{results_path}/result.txt'
        eval_avg_result_file = f'{results_path}/avg_result.txt'

        if os.path.exists(eval_result_file):
            os.remove(eval_result_file)

        if not os.path.exists(results_path):
            os.makedirs(results_path)

    df_result = pd.DataFrame()
    if args.task == 'link_prediction':  # 
        metrics = LINK_PREDICTION_METRICS  # 
        clf = LinkPredictor(args=args, graph=graph, results_path=results_path)  # 
        for method in methods:  # 
            print(f"\n---- {method} link prediction ----")  # 
            result = clf.train_and_evaluate(method, node_subjects, cv_fold=5, n_trials=args.n_trials)
            df_result = pd.concat([df_result, result])  # 

    parent_directory = os.path.dirname(eval_result_file)
    if not os.path.exists(parent_directory):
        os.makedirs(parent_directory)

    df_result.to_csv(eval_result_file, sep='\t', index=False)
    avg_result = df_result.groupby(["method"])[metrics].agg(["mean", "std"]).reset_index()
    for metric in metrics:
        avg_result[(metric, "mean")] = avg_result[(metric, "mean")].round(4)
        avg_result[(metric, "std")] = avg_result[(metric, "std")].round(4)
    avg_result.to_csv(eval_avg_result_file, sep='\t')
    # print("\nMean Performance Metrics:")
    # print(avg_result.to_string(index=False), '\n')


if __name__ == "__main__":
    args = parse_args()
    seed = args.seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    main(args)