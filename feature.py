import numpy as np
import pandas as pd
import tqdm

import matrix
from argument import DATASET, DEVICES_GPU, NUM_NODE


class Feature:
    df = pd.DataFrame()
    n_user = 0
    snode = 0
    tnode = 0
    vec = np.zeros(24)

    def __init__(self, snode, tnode, vec):
        self.snode = snode
        self.tnode = tnode
        self.vec = vec


def extract_features(path, df_target, mtx):
    with open(path, 'w') as f:
        features = []
        pbar = tqdm.tqdm(total=df_target.shape[0])
        for i in range(df_target.shape[0]):
            snode = df_target.iloc[i].sid
            tnode = df_target.iloc[i].tid
            sign = df_target.iloc[i].feedback

            lt_degree = [matrix.get_degree(mtx[0][0], snode), matrix.get_degree(mtx[0][1], snode),
                         matrix.get_degree(mtx[0][2], tnode), matrix.get_degree(mtx[0][3], tnode)]
            lt_degree.extend([lt_degree[0] + lt_degree[1], lt_degree[2] + lt_degree[3], mtx[1][snode, tnode]])
            for j in range(16):
                lt_degree.append(mtx[2][j][snode, tnode])
            lt_degree.append(sign)
            features.append(Feature(snode, tnode, lt_degree))
            f.write(str(snode))
            f.write('\t' + str(tnode) + '\t')
            f.write('\t'.join(map(str, lt_degree)) + '\n')
            pbar.update(1)
        pbar.close()
    return features


# extract features of other nodes from given snode
def extract_features_for_seed(snode, mtx, n_user):
    features = []
    snode_degree_1 = matrix.get_degree(mtx[0][0], snode)
    snode_degree_2 = matrix.get_degree(mtx[0][1], snode)
    for tnode in range(n_user):
        lt_degree = [snode_degree_1, snode_degree_2,
                     matrix.get_degree(mtx[0][2], tnode), matrix.get_degree(mtx[0][3], tnode)]
        lt_degree.extend([lt_degree[0] + lt_degree[1], lt_degree[2] + lt_degree[3], mtx[1][snode, tnode]])
        for j in range(16):
            lt_degree.append(mtx[2][j][snode, tnode])
        features.append(Feature(snode, tnode, lt_degree))
    return features


#  to save memory
def extract_features_for_seed_save_time(snode, mtx, n_user, adj_set):
    features = []
    # [0][0] -> csr pos
    # [0][1] -> csr neg
    # snode가 위치한 행의 데이터 개수(강도)
    snode_degree_1 = matrix.get_degree(mtx[0][0], snode)
    snode_degree_2 = matrix.get_degree(mtx[0][1], snode)
    for tnode in range(n_user):
        if tnode in adj_set:
            # 5개
            lt_degree = [snode_degree_1, snode_degree_2,
                        # [0][2] -> csc pos, [0][3] -> csc neg
                        matrix.get_degree(mtx[0][2], tnode), matrix.get_degree(mtx[0][3], tnode)]
            # [snode 강도, tnode 강도, snode-tnode의 연결 정보, tnode]
            # mtx[1] -> user-item, item-user 모두 연결 된 행렬
            # 3개
            lt_degree.extend([lt_degree[0] + lt_degree[1], lt_degree[2] + lt_degree[3], mtx[1][snode, tnode]])
            for j in range(16):
                # (csr pos-pos, pos-neg, neg-pos, neg-neg) * (R*R, R*R^T, R^T*R, R^T*R^T) -> 16개 행렬
                # 가능한 모든 관계의 정보를 사용하여 강도를 추출
                # 16개
                lt_degree.append(mtx[2][j][snode, tnode])
        else:
            lt_degree = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # 추출 된 강도를 사용해서 새로운 Featrue 생성
        features.append(Feature(snode, tnode, lt_degree))
    return features