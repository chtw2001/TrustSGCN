# Performs different embedding propagation (trustworthy or untrustworthy)

import os
import sys
import time
import math
import random
import subprocess
import logging
import argparse

from collections import defaultdict

import numpy as np
import scipy
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
        
import pickle
from common import DATASET_NUM_DIC

from logistic_function import logistic_embedding_link, logistic_embedding_sign
# mj add
from datetime import datetime
now_=datetime.now().strftime('%y-%m-%d %H:%M:%S')
from tqdm import tqdm 
import dgl
from dgl.data.utils import save_graphs, load_graphs
from dgl.contrib.sampling.sampler import NeighborSampler
from dgl.nn.pytorch import GATConv
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser()
parser.add_argument('--devices', type=str, default='cuda:0', help='Devices')
parser.add_argument('--seed', type=int, default=111, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')

parser.add_argument('--dataset', default='bitcoin_alpha', help='Dataset(bitcoin_alpha, bitcoin_otc, slashdot_aminer, epinions_aminer)')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument("--percent", default=80, help="Sparsity setting(80, 60, 40, 20)")   
parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay (L2 loss on parameters).')

parser.add_argument('--batch_size', type=int, default=300, help='Batch size')
parser.add_argument('--sample_num', type=int, default=30, help='sample_num(bitcoin_alpha: 30, bitcoin_otc: 30, slashdot: 20, epinions: 10)')
# parser.add_argument('--sampler', default=True, help='sampler') 

parser.add_argument('--k', default=1, help='Folder k')
parser.add_argument('--agg', default='mean', choices=['mean'], help='Aggregator choose')
parser.add_argument('--hop', default='2', help='hop')
parser.add_argument("--p_thres", type=float, default=0.98, help="Positive threshold (beta_+). Default is 1.0")
parser.add_argument("--n_thres", type=float, default=0.98, help="Negative threshold (beta_-). Default is 0.6")
# parser.add_argument('--hop_distin', default='learning', help='hop distinguish (no, inverse, learning)')

parser.add_argument('--dim', type=int, default=32, help='Embedding dimension (pos: 32, neg: 32 -> total=64)')
parser.add_argument('--fea_dim', type=int, default=32, help='Feature embedding dimension')
parser.add_argument('--get_dgl', type=str, default='True', choices=['True', 'False'], help='Get DGL graph')
args = parser.parse_args()
PER = args.percent
TASK = args.task
GET_DGL = args.get_dgl
SAMPLER = args.sampler

OUTPUT_DIR = f'./embeddings/trustsgcn-{args.agg}'

if not os.path.exists('embeddings'):
    os.mkdir('embeddings')

if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)
        
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

NEG_LOSS_RATIO = 1
INTERVAL_PRINT = 2
NUM_NODE = DATASET_NUM_DIC[args.dataset]
WEIGHT_DECAY = args.weight_decay
NODE_FEAT_SIZE = args.fea_dim
EMBEDDING_SIZE1 = args.dim
DEVICES = torch.device(args.devices)
LEARNING_RATE = args.lr
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
K = args.k
DATASET=  args.dataset
HOP= int(args.hop)
P_THRESN=args.p_thres
N_THRESN=args.n_thres
# HOP_DISTIN = args.hop_distin
SAMPLE_NUM = args.sample_num

#path
FEA_PATH='./features/{}'.format(DATASET)
RESULT_PATH='/result/{}_{}fold_{}hop_thre-{}-{}_{}lr_{}bat_{}samp_sign_{}.txt'.format(DATASET,K, HOP, P_THRESN, N_THRESN, LEARNING_RATE, BATCH_SIZE, SAMPLE_NUM, PER)


MTX_T1_PATH = FEA_PATH +'/mtxT1-{}-{}-{}-{}-{}hop_{}.npy'.format(DATASET, K, P_THRESN, N_THRESN, HOP, PER)
MTX_T2_PATH = FEA_PATH +'/mtxT2-{}-{}-{}-{}-{}hop_{}.npy'.format(DATASET, K, P_THRESN, N_THRESN, HOP, PER)
MTX_U1_PATH = FEA_PATH +'/mtxU1-{}-{}-{}-{}-{}hop_{}.npy'.format(DATASET, K, P_THRESN, N_THRESN, HOP, PER)
MTX_U2_PATH = FEA_PATH +'/mtxU2-{}-{}-{}-{}-{}hop_{}.npy'.format(DATASET, K, P_THRESN, N_THRESN, HOP, PER)
DGLGRAPH_T1_PATH= FEA_PATH + '/graphT1{}_u{}_{}_{}_{}_{}.npy'.format(DATASET, K, HOP, P_THRESN, N_THRESN,PER)
DGLGRAPH_T2_PATH= FEA_PATH + '/graphT2{}_u{}_{}_{}_{}_{}.npy'.format(DATASET, K, HOP, P_THRESN, N_THRESN,PER)
DGLGRAPH_U1_PATH= FEA_PATH + '/graphU1{}_u{}_{}_{}_{}_{}.npy'.format(DATASET, K, HOP, P_THRESN, N_THRESN,PER)
DGLGRAPH_U2_PATH= FEA_PATH + '/graphU2{}_u{}_{}_{}_{}_{}.npy'.format(DATASET, K, HOP, P_THRESN, N_THRESN,PER)

class Encoder(nn.Module):
    def __init__(self, features_pos, features_neg, feature_dim, embed_dim, MLG, aggs):
        super(Encoder, self).__init__()

        self.features_pos = features_pos
        self.features_neg = features_neg
        self.feat_dim = feature_dim

        self.MLG = MLG
        self.aggs = aggs

        self.embed_dim = embed_dim
        # self.aggs[i] -> MeanAggregator
        for i, agg in enumerate(self.aggs):
            self.add_module('agg_{}'.format(i), agg)
            self.aggs[i] = agg.to(DEVICES)

        self.pos_nonlinear_layer = nn.Sequential(
                nn.Linear(2 * feature_dim, feature_dim),
                nn.Tanh(),
                nn.Linear(feature_dim, int(feature_dim))
        )

        self.neg_nonlinear_layer = nn.Sequential(
                nn.Linear(2 * feature_dim, feature_dim),
                nn.Tanh(),
                nn.Linear(feature_dim, int(feature_dim))
        )



    def forward(self, nodes):
        # eval: nodes -> random nodes
        if not isinstance(nodes, list) and nodes.is_cuda:
            nodes = nodes.data.cpu().numpy().tolist()
 
        # 모든 그래프별 node에 대해서 positive, negative embedding을 각각 계산한 결과
        neigh_feats_all= [agg(nodes, adj, ind) for adj, agg, ind in zip(self.MLG, self.aggs, range(len(self.MLG)))]
        final_pos = neigh_feats_all[0][0] + neigh_feats_all[1][0] + neigh_feats_all[2][0] + neigh_feats_all[3][0]
        final_neg = neigh_feats_all[0][1] + neigh_feats_all[1][1] + neigh_feats_all[2][1] + neigh_feats_all[3][1]

        final_pos = torch.cat([self.features_pos(torch.LongTensor(nodes).to(DEVICES)), final_pos], 1)
        final_neg = torch.cat([self.features_neg(torch.LongTensor(nodes).to(DEVICES)), final_neg], 1)
       
        combined_pos = self.pos_nonlinear_layer(final_pos)
        combined_neg = self.neg_nonlinear_layer(final_neg)
        combined = torch.cat([combined_pos, combined_neg], dim=1)

        # (NUM_NODE, 64)
        return combined

class EncoderP(nn.Module):
    def __init__(self, features_pos, features_neg, feature_dim, embed_dim, MLG, aggs):
        super(EncoderP, self).__init__()

        self.features_pos = features_pos
        self.features_neg = features_neg
        self.feat_dim = feature_dim
        self.MLG = MLG
        self.aggs = aggs

        self.embed_dim = embed_dim
        for i, agg in enumerate(self.aggs):
            self.add_module('agg_{}'.format(i), agg)
            self.aggs[i] = agg.to(DEVICES)

        self.pos_nonlinear_layer = nn.Sequential(
                nn.Linear((2 * len(MLG)) * feature_dim, feature_dim),
                nn.Tanh(),
                nn.Linear( feature_dim, feature_dim)
        )

 
    def forward(self, nodes):
        if not isinstance(nodes, list) and nodes.is_cuda:
            nodes = nodes.data.cpu().numpy().tolist()

        neigh_feats_all= [agg(nodes, adj, ind) for adj, agg, ind in zip(self.MLG, self.aggs, range(len(self.MLG)))]
        final_pos = torch.cat([self.features_pos(torch.LongTensor(nodes).to(DEVICES)), neigh_feats_all[0][0]], 1)
        combined_pos = self.pos_nonlinear_layer(final_pos)
         
        return combined_pos

class EncoderN(nn.Module):
    def __init__(self, features_pos, features_neg, feature_dim, embed_dim, MLG, aggs):
        super(EncoderN, self).__init__()

        self.features_pos = features_pos
        self.features_neg = features_neg
        self.feat_dim = feature_dim
        self.MLG = MLG
        self.aggs = aggs

        self.embed_dim = embed_dim
        for i, agg in enumerate(self.aggs):
            self.add_module('agg_{}'.format(i), agg)
            self.aggs[i] = agg.to(DEVICES)

        self.neg_nonlinear_layer = nn.Sequential(
                nn.Linear((2 * len(MLG)) * feature_dim, feature_dim),
                nn.Tanh(),
                nn.Linear( feature_dim, feature_dim)
        )


    def forward(self, nodes):
        if not isinstance(nodes, list) and nodes.is_cuda:
            nodes = nodes.data.cpu().numpy().tolist()        
        neigh_feats_all= [agg(nodes, adj, ind) for adj, agg, ind in zip(self.MLG, self.aggs, range(len(self.MLG)))]

        final_neg = torch.cat([self.features_neg(torch.LongTensor(nodes).to(DEVICES)), neigh_feats_all[0][1]], 1)

        combined_neg = self.neg_nonlinear_layer(final_neg)
     
        return combined_neg

class MeanAggregator(nn.Module):
    # MeanAggregator인데 aggregator 메서드에서 sum만 하네
    # 평균 집계
    def __init__(self, features_pos, features_neg, in_dim, out_dim, percent):
        super(MeanAggregator, self).__init__()

        self.features_pos = features_pos
        self.features_neg = features_neg
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.out_linear_layer = nn.Sequential( 
            nn.Linear(self.in_dim*2, self.out_dim),
            nn.Tanh(),
            nn.Linear(self.out_dim, self.out_dim)
        ).to(DEVICES)
        # HOP = 2
        # 단위 벡터로 정규화한 값을 hop이 2이상인 노드에 곱하는 가중치
        self.alpha=torch.nn.Embedding(HOP, 1, max_norm=1.0).to(DEVICES)
      
        nn.init.ones_(self.alpha.weight)
       
        self.percent = percent


    def message(self, edges): 
      
        # src -> out node에 해당
        em_p = edges.src['node_emb_p'] 
        em_n = edges.src['node_emb_n'] 

        # 2 hop 이상인 노드에 가중치 alpha를 적용
        mask = ((edges.data['hop']-1)>0)
       
        em_p[mask] = edges.src['node_emb_p'][mask] * self.alpha(edges.data['hop']-1)[mask].view(-1,1)
        em_n[mask] = edges.src['node_emb_n'][mask] * self.alpha(edges.data['hop']-1)[mask].view(-1,1)
       
        em_pp= em_p * self.percent[0] # Trust Sign 1, Trust Unsign 0
        em_pn= em_p * self.percent[1] # Trust Sign 0, Trust Unsign 1
        em_np= em_n * self.percent[2] # Trust Sign 0, Trust Unsign 1
        em_nn= em_n * self.percent[3] # Trust Sign 1, Trust Unsign 0
 
        # 노드 강도로 정규화
        # 왜 np는 emb_p에 들어가고, pn은 emb_n에 들어가지?
        emb_p = (em_pp + em_np) / edges.src['out_degree']
        emb_n = (em_pn + em_nn) / edges.src['out_degree']

        return {'m_p' : emb_p, 'm_n' : emb_n}

    def aggregator(self,nodes):
        # out node가 받은 message를 더해서 dictionary로 반환
        return {'node_emb_p2': nodes.mailbox['m_p'].sum(1), 'node_emb_n2': nodes.mailbox['m_n'].sum(1)} #degree normalization 할때는 sum 해야함



    def forward(self, batch_nodes, MLG, ind):
        MLG.readonly(True)
        neightbor_sampler = NeighborSampler( MLG , len(batch_nodes) , expand_factor=SAMPLE_NUM , neighbor_type='in',  seed_nodes=batch_nodes)
    
        for sub_sample in neightbor_sampler:
            break
        sub_sample.copy_from_parent()

        # layers[0] -> seed node
        sub_sample.layers[0].data['node_emb_p'] =  self.features_pos(sub_sample.layers[0].data['node_idx'].to(DEVICES))
        sub_sample.layers[0].data['node_emb_n'] =  self.features_neg(sub_sample.layers[0].data['node_idx'].to(DEVICES))
        # 1hop의 out node 강도. 정규화 용도
        sub_sample.layers[0].data['out_degree'] = sub_sample.layer_out_degree(0).type(torch.FloatTensor).unsqueeze(1).to(DEVICES) # degree normalization
        # message_func -> 이웃 노드로부터 메시지 생성
        # reduce_func -> 메시지를 전파하고 업데이트
        sub_sample.block_compute(block_id=0,  message_func = self.message, reduce_func = self.aggregator)
       
        # layers[0] -> 1hop neighbor node
        sub_sample.layers[1].data['node_emb_p'] = sub_sample.layers[1].data['node_emb_p2'] 
        sub_sample.layers[1].data['node_emb_n'] = sub_sample.layers[1].data['node_emb_n2'] 

        sub_sample.layers[1].data.pop('node_emb_p2')
        sub_sample.layers[1].data.pop('node_emb_n2')

        final_embeddings_pos = sub_sample.layers[1].data['node_emb_p']#[batch_nodes]
        final_embeddings_neg = sub_sample.layers[1].data['node_emb_n']#[batch_nodes]

        return final_embeddings_pos, final_embeddings_neg

class TrustSGCN(nn.Module):

    def __init__(self, enc):
        super(TrustSGCN, self).__init__()
        self.enc = enc
        self.score_function1 = nn.Sequential(
            nn.Linear(EMBEDDING_SIZE1*2, 1),
            nn.Sigmoid()
        )
        self.score_function2 = nn.Sequential(
            nn.Linear(EMBEDDING_SIZE1*2, 1),
            nn.Sigmoid()
        )
 

    def forward(self, nodes):
        # (NUM_NODE, 64)
        embeds = self.enc(nodes)
        return embeds

    def criterion(self, nodes, pos_neighbors, neg_neighbors, adj_lists_out_p, adj_lists_out_n): 
        # feedback이 1인 node list       
        pos_neighbors_list = [set.union(pos_neighbors[i]) for i in nodes]
        # feedback이 -1인 node list
        neg_neighbors_list = [set.union(neg_neighbors[i]) for i in nodes]

        unique_nodes_list = list(set.union(*pos_neighbors_list).union(*neg_neighbors_list).union(nodes))
        # key: unique node, value: idx
        unique_nodes_dict = {n: i for i, n in enumerate(unique_nodes_list)}
        nodes_embs = self.enc(unique_nodes_list)

        sign_loss = 0
        status_loss = 0

        for index, node in enumerate(nodes):
            a_emb = nodes_embs[unique_nodes_dict[node], :] 
            # neighbor의 idx
            pos_neigs = list([unique_nodes_dict[i] for i in pos_neighbors[node]]) 
            neg_neigs = list([unique_nodes_dict[i] for i in neg_neighbors[node]]) 
            pos_num = len(pos_neigs)
            neg_num = len(neg_neigs)

            sta_pos_neighs_out = list([unique_nodes_dict[i] for i in adj_lists_out_p[node]])
            sta_neg_neighs_out = list([unique_nodes_dict[i] for i in adj_lists_out_n[node]])          

            if pos_num > 0:
                # 이웃들의 embedding
                pos_neig_embs = nodes_embs[pos_neigs, :] 

                # sign loss: entropy
                # (pos_num, 64) * (64,) -> (pos_num,)
                # 이웃 노드간 얼마나 닮아있는지 log함수를 취하여 계산. 1에 가까울수록 작음
                sign_loss += F.binary_cross_entropy_with_logits(torch.einsum("nj,j->n", [pos_neig_embs, a_emb]), torch.ones(pos_num).to(DEVICES))     

              
            if neg_num > 0:
                neg_neig_embs = nodes_embs[neg_neigs, :]

                # sign loss: entropy
                sign_loss += F.binary_cross_entropy_with_logits(torch.einsum("nj,j->n", [neg_neig_embs, a_emb]), torch.zeros(neg_num).to(DEVICES)) 

        return sign_loss 
            
def load_data1(filename=''):
    mtx_T1 = np.load(MTX_T1_PATH, allow_pickle=True)
    mtx_T2 = np.load(MTX_T2_PATH, allow_pickle=True)
    mtx_U1 = np.load(MTX_U1_PATH, allow_pickle=True)
    mtx_U2 = np.load(MTX_U2_PATH, allow_pickle=True)
    
    return mtx_T1, mtx_T2, mtx_U1, mtx_U2  


def load_data2(filename=''):
    # 양방향
    adj_list_pos = defaultdict(set)
    # out: source -> target
    adj_list_out_pos = defaultdict(set)
    # in: target -> source
    adj_list_in_pos = defaultdict(set)
    adj_list_neg = defaultdict(set)
    adj_list_out_neg = defaultdict(set)
    adj_list_in_neg = defaultdict(set)
    # 단방향. 사실 pos
    adj_list_out_unsign = defaultdict(set)

    with open(filename) as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            person1 = int(info[0]) # source
            person2 = int(info[1]) # target
            v = int(info[2])       # feedback
            adj_list_out_unsign[person1].add(person2)  

            if v == 1:
                # 양방향 edge
                adj_list_pos[person1].add(person2)
                adj_list_pos[person2].add(person1)

                adj_list_out_pos[person1].add(person2)
                adj_list_in_pos[person2].add(person1) 
            else:
                adj_list_neg[person1].add(person2)
                adj_list_neg[person2].add(person1)

                adj_list_out_neg[person1].add(person2)
                adj_list_in_neg[person2].add(person1)

    return adj_list_pos, adj_list_out_pos, adj_list_in_pos, adj_list_neg, adj_list_out_neg, adj_list_in_neg, adj_list_out_unsign

def run(dataset, k):

    with open(OUTPUT_DIR+RESULT_PATH, 'a') as res:
        res.write("time: "+ str(now_) +"\n")
        res.write("DATASET: "+ DATASET +"\n")
        res.write("LEARNING_RATE: "+ str(LEARNING_RATE) +"\n")
        res.write("WEIGHT_DECAY: "+ str(WEIGHT_DECAY) +"\n")
        res.write("NODE_FEAT_SIZE: "+ str(NODE_FEAT_SIZE) +"\n")
        res.write("EMBEDDING_SIZE: "+str( EMBEDDING_SIZE1) +"\n")
        res.write("BATCH_SIZE: "+ str(BATCH_SIZE) +"\n")
        res.write("EPOCHS: "+ str(EPOCHS) +"\n")
        res.write("HOP: "+ str(HOP) +"\n")
        # res.write("HOP_DISTIN: "+ str(HOP_DISTIN) +"\n")
        res.write("task: "+ TASK +"\n")
    
        res.write("EPOCH\t" +  "LOSS\t\t" + "pos_ratio\t\t"+ "accuracy\t\t" + "f1_score\t\t"+ "microf1_score\t\t" + "macrof1_score\t\t"+ "auc_score\n")
    res.close()

    num_nodes = DATASET_NUM_DIC[dataset] 

    filename = './experiment-data/{}/{}_u{}_{}.train'.format(dataset,dataset, k, PER)
    
    # undirect: 양방향
    # outdirect: source -> target
    # indirect: target -> source
    undirect_pos, outdirect_pos, indirect_pos, undirect_neg, outdirect_neg, indirect_neg, outdirect_unsign = load_data2(filename) # pos_all, pos out, pos in, neg_all, neg out, neg in, posnegall
    # (Trust, sign=1), (Trust, sign=-1), (UnTrust, sign=1), (UnTrust, sign=-1)
    mtx_T1, mtx_T2, mtx_U1, mtx_U2 = load_data1(filename)  
    print(k, dataset, 'data load!')

    graphT1 = dgl.DGLGraph()
    graphT2 = dgl.DGLGraph()
    graphU1 = dgl.DGLGraph()
    graphU2 = dgl.DGLGraph()

    if GET_DGL=='True':
        # source -> target으로 edge를 추가하고 해당 edge에 sign, hop 정보를 추가
        graphT1.add_nodes(num_nodes)
        graphT1.ndata['node_idx'] = torch.tensor(list(range(num_nodes)))
        # source, target, sign, hop
        p1s1, p2s1, signs1, hop1 = zip(*mtx_T1)
        for p1, p2 in tqdm(zip(p1s1, p2s1)): #17분
            graphT1.add_edges(int(p1), int(p2))
        graphT1.edata['sign'] = torch.IntTensor(list(signs1))
        graphT1.edata['hop'] = torch.IntTensor(list(hop1))
        save_graphs(DGLGRAPH_T1_PATH, graphT1)

    
        graphT2.add_nodes(num_nodes)
        graphT2.ndata['node_idx'] = torch.tensor(list(range(num_nodes)))
        p1s2, p2s2, signs2, hop2 = zip(*mtx_T2)
        for p1, p2 in tqdm(zip(p1s2, p2s2)): #17분
            graphT2.add_edges(int(p1), int(p2))
        graphT2.edata['sign'] = torch.IntTensor(list(signs2))
        graphT2.edata['hop'] = torch.IntTensor(list(hop2))
        save_graphs(DGLGRAPH_T2_PATH, graphT2)

   
        graphU1.add_nodes(num_nodes)
        graphU1.ndata['node_idx'] = torch.tensor(list(range(num_nodes)))
        p1s3, p2s3, signs3, hop3 = zip(*mtx_U1)
        for p1, p2 in tqdm(zip(p1s3, p2s3)): #17분
            graphU1.add_edges(int(p1), int(p2))
        graphU1.edata['sign'] = torch.IntTensor(list(signs3))
        graphU1.edata['hop'] = torch.IntTensor(list(hop3))
        save_graphs(DGLGRAPH_U1_PATH, graphU1)
  

        graphU2.add_nodes(num_nodes)
        graphU2.ndata['node_idx'] = torch.tensor(list(range(num_nodes)))
        p1s4, p2s4, signs4, hop4 = zip(*mtx_U2)
        for p1, p2 in tqdm(zip(p1s4, p2s4)): #17분
            graphU2.add_edges(int(p1), int(p2))
        graphU2.edata['sign'] = torch.IntTensor(list(signs4))
        graphU2.edata['hop'] = torch.IntTensor(list(hop4))
        save_graphs(DGLGRAPH_U2_PATH, graphU2)
    
    
    else:
        print("load dglgraph!")
        mlg_graph = load_graphs(DGLGRAPH_T1_PATH)
        graphT1=mlg_graph[0][0]

        mlg_graph = load_graphs(DGLGRAPH_T2_PATH)
        graphT2=mlg_graph[0][0]

        mlg_graph = load_graphs(DGLGRAPH_U1_PATH)
        graphU1=mlg_graph[0][0]

        mlg_graph = load_graphs(DGLGRAPH_U2_PATH)
        graphU2=mlg_graph[0][0]

        
    graphT1.edata['hop'] = graphT1.edata['hop'].to(DEVICES) 
    graphT2.edata['hop'] = graphT2.edata['hop'].to(DEVICES) 
    graphU1.edata['hop'] = graphU1.edata['hop'].to(DEVICES) 
    graphU2.edata['hop'] = graphU2.edata['hop'].to(DEVICES)

    
    if DATASET=="wikiRfA_pos_neg":
        # pre_analysis는 어떤 비율일까?
        pre_analysis = [(0.71, 0.29), (0.71, 0.29), (0.71, 0.29), (0.86, 0.14)] 
        percent = [(1.0 ,0.0 ,0.0 ,1.0 ), (0.0 ,1.0 ,1.0 ,0.0 ), (pre_analysis[0][0], pre_analysis[0][1], pre_analysis[2][0], pre_analysis[2][1]), (pre_analysis[1][0], pre_analysis[1][1], pre_analysis[3][0], pre_analysis[3][1])]
        
    elif DATASET=="slashdot_aminer":
        pre_analysis = [(0.96, 0.04), (0.71, 0.29), (0.71, 0.29), (0.9, 0.1)] #undirect
        percent = [(1.0 ,0.0 ,0.0 ,1.0 ), (0.0 ,1.0 ,1.0 ,0.0 ), (pre_analysis[0][0], pre_analysis[0][1], pre_analysis[2][0], pre_analysis[2][1]), (pre_analysis[1][0], pre_analysis[1][1], pre_analysis[3][0], pre_analysis[3][1])]

    elif DATASET=="epinions_aminer":
        pre_analysis = [(0.82, 0.18), (0.51, 0.49), (0.51, 0.49), (0.79, 0.21)] #undirect
        percent = [(1.0 ,0.0 ,0.0 ,1.0 ), (0.0 ,1.0 ,1.0 ,0.0 ), (pre_analysis[0][0], pre_analysis[0][1], pre_analysis[2][0], pre_analysis[2][1]), (pre_analysis[1][0], pre_analysis[1][1], pre_analysis[3][0], pre_analysis[3][1])]
            
    elif DATASET=="bitcoin_otc":
        pre_analysis = [(0.86, 0.14), (0.42, 0.58), (0.42, 0.58), (0.94, 0.06)] 
        percent = [(1.0 ,0.0 ,0.0 ,1.0 ), (0.0 ,1.0 ,1.0 ,0.0 ), (pre_analysis[0][0], pre_analysis[0][1], pre_analysis[2][0], pre_analysis[2][1]), (pre_analysis[1][0], pre_analysis[1][1], pre_analysis[3][0], pre_analysis[3][1])]

    elif DATASET=="bitcoin_alpha":
        pre_analysis = [(0.85, 0.15), (0.63, 0.37), (0.63, 0.37), (0.93, 0.07)] #undirect
        percent = [(1.0 ,0.0 ,0.0 ,1.0 ), (0.0 ,1.0 ,1.0 ,0.0 ), (pre_analysis[0][0], pre_analysis[0][1], pre_analysis[2][0], pre_analysis[2][1]), (pre_analysis[1][0], pre_analysis[1][1], pre_analysis[3][0], pre_analysis[3][1])]

    features_pos = nn.Embedding(num_nodes, NODE_FEAT_SIZE)
    features_pos.weight.requires_grad = True
    features_pos = features_pos.to(DEVICES)

    features_neg = nn.Embedding(num_nodes, NODE_FEAT_SIZE)
    features_neg.weight.requires_grad = True
    features_neg = features_neg.to(DEVICES)

    MLG = [graphT1, graphT2, graphU1, graphU2]
    aggregator = MeanAggregator


    aggs = [aggregator(features_pos,features_neg, NODE_FEAT_SIZE, NODE_FEAT_SIZE, percent[i]) for i in range(len(MLG))]
    enc1 = Encoder(features_pos,features_neg, NODE_FEAT_SIZE, EMBEDDING_SIZE1, MLG, aggs)
    model = TrustSGCN(enc1)


    # for layer2
    '''enc1_pos = EncoderP(features_pos,features_neg, NODE_FEAT_SIZE, EMBEDDING_SIZE1, MLG, aggs)
    enc1_neg = EncoderN(features_pos,features_neg, NODE_FEAT_SIZE, EMBEDDING_SIZE1, MLG, aggs)
    # enc1_pos = enc1_pos.to(DEVICES)
    # enc1_neg = enc1_neg.to(DEVICES)


    aggs2 = [aggregator(enc1_pos, enc1_neg, EMBEDDING_SIZE1, EMBEDDING_SIZE1, num_nodes, adj) for adj in MLG]
    enc2 = Encoder(enc1_pos, enc1_neg, EMBEDDING_SIZE1, EMBEDDING_SIZE1, MLG, aggs2)

    model = TrustSGCN(enc2)'''

    model = model.to(DEVICES)

    print(model.train())

    #for layer1
    # gradient update가 필요한 가중치만 업데이트
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    #for layer2
    '''optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, list(model.parameters()) + list(enc1_pos.parameters()) + list(enc1_neg.parameters()) + list(enc2.parameters())\
                                        + list(features_pos.parameters()) + list(features_neg.parameters())),
                                 lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY
                                 )'''

    '''optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, list(model.parameters()) \
                                        + list(features_pos.parameters()) + list(features_neg.parameters())),
                                 lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY
                                 )'''


    time_result = []
    for epoch in range(EPOCHS + 2):
        total_loss = []
        # INTERVAL_PRINT = 2
        if epoch % INTERVAL_PRINT == 0:
            model.eval()
            all_embedding = np.zeros((NUM_NODE, EMBEDDING_SIZE1*2))
            for i in range(0, NUM_NODE, BATCH_SIZE):
                begin_index = i
                end_index = i + BATCH_SIZE if i + BATCH_SIZE < NUM_NODE else NUM_NODE
                values = np.arange(begin_index, end_index)
                # (NUM_NODE, 64). 절반은 pos, 나머지는 neg
                embed = model.forward(values.tolist())
                embed = embed.data.cpu().numpy()
                all_embedding[begin_index: end_index] = embed

            model.train()

        time1 = time.time()
        nodes_pku = np.random.permutation(NUM_NODE).tolist()
        for batch in range(NUM_NODE // BATCH_SIZE):
            optimizer.zero_grad()
            b_index = batch * BATCH_SIZE
            e_index = (batch + 1) * BATCH_SIZE
            nodes = nodes_pku[b_index:e_index]

            # tri loss
            # 이웃 노드간 유사도로 loss 계산
            loss = model.criterion(
                # undirect: 양방향
                # outdirect: source -> target
                nodes, undirect_pos, undirect_neg, outdirect_pos, outdirect_neg
            )

    
            total_loss.append(loss.data.cpu().numpy())

            loss.backward()
            optimizer.step()
        time_result.append(time.time() - time1)
        if epoch%20 == 0:
            print(f'------------epoch: {epoch + 1}, time: {np.array(time_result).mean()}----------------')
        

        fpath = os.path.join(OUTPUT_DIR, 'embedding-{}-{}-{}.npy'.format(dataset, k, str(epoch)))
        # node별 embedding 저장
        np.save(fpath, all_embedding)
        
        if epoch%INTERVAL_PRINT==0:
            pos_ratio, accuracy, f1_score0, f1_ma, f1_mi, auc_score = logistic_embedding_sign(k=k, dataset=dataset, epoch=epoch, dirname=OUTPUT_DIR, per=PER)
            
            with open(OUTPUT_DIR+RESULT_PATH, 'a') as res:
                    res.write(str(epoch) + "\t" + str(np.mean(total_loss)) + "\t" + str(pos_ratio) + "\t" + str(accuracy) +  "\t"+ str(f1_score0) + "\t"+  str(f1_mi) + "\t"+ str(f1_ma) + "\t" + str(auc_score) +"\n")
                
            print(f'epoch: {epoch}, loss: {np.mean(total_loss)}')
                

def main():
    print('NUM_NODE', NUM_NODE)
    print('WEIGHT_DECAY', WEIGHT_DECAY)
    print('NODE_FEAT_SIZE', NODE_FEAT_SIZE)
    print('EMBEDDING_SIZE1', EMBEDDING_SIZE1)
    print('LEARNING_RATE', LEARNING_RATE)
    print('BATCH_SIZE', BATCH_SIZE)
    print('EPOCHS', EPOCHS)
    print('TASK', TASK)
    dataset = args.dataset
    run(dataset=dataset, k=K)


if __name__ == "__main__":
    main()

