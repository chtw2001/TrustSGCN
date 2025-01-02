from datetime import datetime
import time
from collections import defaultdict
import numpy as np
from tqdm import tqdm 
import pickle
import pandas as pd

from argument import RANDOM_SEED, NUM_NODE, K, DATASET, HOP, FUNCTION, P_THRESN, N_THRESN, PER

from trustworthiness import extract_features, process, predict_FExtra_scores_save_time 

from count_tri import cntTriangle

np.random.seed(RANDOM_SEED)
now_=datetime.now().strftime('%y-%m-%d %H:%M:%S')

def set_subgraph(flag, filename, max_hop, TWO_HOP_SUB, dic_path, THR_HOP_SUB):
    # graph에서 연결 된 node를 최대 3hop까지 연결하고 dictionary 반환
    if flag:
        print("Set subgraph.........")
        adj_lists_all = defaultdict(list)
        adj_lists_1hop = set()
        adj_lists_2hop = defaultdict(list)

        with open(filename) as fp:
    
            for i, line in enumerate(fp):
                info = line.strip().split()
                person1 = int(info[0]) # from
                person2 = int(info[1]) # to
                sign = int(info[2])# sign
                value1=[person1, sign, 1] # from node, sign, hop
                value2=[person2, sign, 1] # to node, sign, hop

                # 1 hop으로 연결 된 node 삽입
                adj_lists_all[person1].append(value2)
                adj_lists_all[person2].append(value1)
            
            time1 = time.time()

            # 2hop 
            if max_hop == 2:
                for p1 in tqdm(adj_lists_all.keys(), bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}'):
                    # 중복 없이
                    adj_lists_all[p1][:] = list(set(map(tuple, adj_lists_all[p1][:])))
                    final_adj_lists_all = adj_lists_all[p1][:] 
                    for p2,sign,hop in final_adj_lists_all:
                    
                        if hop ==1:    
                            for hp, hsign, hhop in adj_lists_all[p2]:
                                if hp != p1 and hhop==1:
                                    # 양방향 모두 final_adj_lists_all에 삽입, hop은 2로
                                    final_adj_lists_all.append([hp, hsign*sign, 2])                    
             
                    # 양방향으로 연결 된 node는 sign을 곱하고 hop을 2로 설정
                    adj_lists_all[p1] = final_adj_lists_all[:]
            
            #3hop 
            # hop2와 동일한 동작
            if max_hop == 3:
                with open(TWO_HOP_SUB, 'rb') as fr:
                    adj_lists_2hop = pickle.load(fr)

                for me in tqdm(adj_lists_2hop.keys(), bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}'):
                    adj_lists_2hop[me][:] = list(set(map(tuple, adj_lists_2hop[me][:])))
                    final_adj_lists_all = adj_lists_2hop[me][:] 
                    for frieds, fsign, fhop in final_adj_lists_all: 
                        if(fhop==2):
                            for ffriends, ffsign, ffhop in adj_lists_all[frieds]: 
                                if(ffriends != me  and ffhop ==  1): 
                                    temp = [ffriends, ffsign*fsign , 3]
                                    final_adj_lists_all.append(temp)                    
                    adj_lists_all[me] = list(set(map(tuple, final_adj_lists_all[:]))) 
                    adj_lists_all[me] = final_adj_lists_all[:] 
            

            adj_lists_all_hop_cleaning=defaultdict(list)
            for src in tqdm(adj_lists_all.keys(), bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}'):
                onehopset=list()
                # hop 1부터 오름차순 정렬
                a = sorted(adj_lists_all[src], key=lambda x: x[-1])
                for frd, sign, hop in a:
                    if hop==1: 
                        onehopset.append([src,frd])
                        adj_lists_all_hop_cleaning[src].append([frd,sign,hop])
                    else:
                        if [src,frd] in onehopset: pass
                        else: adj_lists_all_hop_cleaning[src].append([frd,sign,hop])
            running_time = time.time() - time1
            print("time: ", running_time)


        with open(dic_path, 'wb')  as fw:
            pickle.dump(adj_lists_all_hop_cleaning, fw)

    else:
        print("Load subgraph.........")


        with open(dic_path, 'rb') as fr:
            adj_lists_all_hop_cleaning = pickle.load(fr)

    # key: source, value: [destination, sign, hop]
    return adj_lists_all_hop_cleaning


            

def init(FUNCTION, P_THRESN, N_THRESN):
    TRAIN_PATH = './experiment-data/{}/{}_u{}_{}.train'.format(DATASET, DATASET, K, PER)
  
    FEA_PATH='./features/{}'.format(DATASET)
    Count_UT= FEA_PATH +'/CountUT-{}-{}-{}-{}-{}hop_{}.txt'.format(DATASET, K, P_THRESN, N_THRESN, HOP, PER)

    MTX_T1_PATH = FEA_PATH +'/mtxT1-{}-{}-{}-{}-{}hop_{}.npy'.format(DATASET, K, P_THRESN, N_THRESN, HOP, PER)
    MTX_T2_PATH = FEA_PATH +'/mtxT2-{}-{}-{}-{}-{}hop_{}.npy'.format(DATASET, K, P_THRESN, N_THRESN, HOP, PER)
    MTX_U1_PATH = FEA_PATH +'/mtxU1-{}-{}-{}-{}-{}hop_{}.npy'.format(DATASET, K, P_THRESN, N_THRESN, HOP, PER)
    MTX_U2_PATH = FEA_PATH +'/mtxU2-{}-{}-{}-{}-{}hop_{}.npy'.format(DATASET, K, P_THRESN, N_THRESN, HOP, PER)

    
    SUBGRAPH_DIC_PATH = FEA_PATH + '/{}_u{}_{}_{}-subgraphDic.pickle'.format(DATASET, K, HOP,PER)
    TWO_HOP_SUB_PATH = FEA_PATH + '/{}_u{}_{}_{}-subgraphDic.pickle'.format(DATASET, K, 2, PER)
    THR_HOP_SUB_PATH = FEA_PATH + '/{}_u{}_{}_{}-subgraphDic.pickle'.format(DATASET, K, 3, PER)
    
    print("experimnet-dataset: ", DATASET)
    print("function: ", FUNCTION)

    if FUNCTION == "countTRI":
        print("========count triangle========")
        print('DATASET', DATASET)
        time1 = time.time()
        cntTriangle(DATASET)
        running_time = time.time() - time1
        print("time: ", running_time)
       
        return
    else:
        print("pass count triangle")


    if FUNCTION == "extract":
        df_train, features_train, mtx = extract_features(True, FEA_PATH, NUM_NODE)
        return
    else:
        # original train dataframe
        # features_train -> (train_num, Feature class)
        # mtx -> [(csr pos, csr neg, csc pos, csc neg), (user-item, item-user 모두 연결 된 행렬), (csr pos-pos, pos-neg, neg-pos, neg-neg R*R, R*R^T, R^T*R, R^T*R^T)]
        df_train, features_train, mtx = extract_features(False, FEA_PATH, NUM_NODE)

    if FUNCTION == "setsubgraph":
        dic = set_subgraph(flag=True, filename=TRAIN_PATH, max_hop=HOP, TWO_HOP_SUB=TWO_HOP_SUB_PATH, dic_path=SUBGRAPH_DIC_PATH, THR_HOP_SUB = THR_HOP_SUB_PATH) # TRAIN DATASET으로 서브 그래프  #, lists_pos, lists_neg
        return
    else:
        # key: source, value: [destination, sign, hop]
        dic = set_subgraph(flag=False, filename=TRAIN_PATH, max_hop=HOP, TWO_HOP_SUB=TWO_HOP_SUB_PATH, dic_path=SUBGRAPH_DIC_PATH, THR_HOP_SUB = THR_HOP_SUB_PATH) # TRAIN DATASET으로 서브 그래프  #, lists_pos, lists_neg
    

    if FUNCTION == "predict":
        mtx = predict_FExtra_scores_save_time(True, FEA_PATH, features_train, mtx, NUM_NODE,dic)#트레인으로 학습한 feature를  기반으로 test데이터의 featrue 점수 계산
        return
    else:
        # mtx -> [csc pos, csc neg] 만 남음
        mtx = predict_FExtra_scores_save_time(False, FEA_PATH, features_train, mtx, NUM_NODE,dic) 
    
    if FUNCTION == "setproMTX":
        # Untrustworthy_percent 저장하는 역할
        # FUNCTION 기본 값은 setproMTX
        #여기서  FExtra가 인풋으로 들어감 : FEA_PATH
        # P_THRESN, N_THRESN -> 0.98, 0.98
        lists_T1, lists_T2, lists_U1, lists_U2 =process(FEA_PATH, df_train, NUM_NODE, [P_THRESN, N_THRESN], dic)
        Untrustworthy_percent =  (len(lists_U1) + len(lists_U2)) / (len(lists_T1) + len(lists_T2)+ len(lists_U1) + len(lists_U2))
        with open(Count_UT, 'a') as res:
            res.write("DATASET: "+ DATASET +"\n")
            res.write("UntrustWorthy 비율: "+ str(Untrustworthy_percent) +"\n")
        res.close()
        
        np.save(MTX_T1_PATH, lists_T1)
        np.save(MTX_T2_PATH, lists_T2)
        np.save(MTX_U1_PATH, lists_U1)
        np.save(MTX_U2_PATH, lists_U2)

        return
    
    else:
        lists_T1 = np.load(MTX_T1_PATH)
        lists_T2 = np.load(MTX_T2_PATH)
        lists_U1 = np.load(MTX_U1_PATH)
        lists_U2 = np.load(MTX_U2_PATH)
        Untrustworthy_percent =  (len(lists_U1) + len(lists_U2)) / (len(lists_T1) + len(lists_T2)+ len(lists_U1) + len(lists_U2))
        with open(Count_UT, 'a') as res:
            res.write("DATASET: "+ DATASET +"\n")
            res.write("UntrustWorthy 비율: "+ str(Untrustworthy_percent) +"\n")
        res.close()
        print("finish!")

if __name__ == "__main__":
    init(FUNCTION, P_THRESN, N_THRESN)
    
