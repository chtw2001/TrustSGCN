import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix, csc_matrix


def df_to_csr_matrix(df, n_user):
    mtx_csr = csr_matrix((df.feedback, (df.sid, df.tid)), shape=(n_user, n_user))
    return mtx_csr


def df_to_csc_matrix(df, n_user):
    mtx_csc = csc_matrix((df.feedback, (df.sid, df.tid)), shape=(n_user, n_user))
    return mtx_csc


def set_mtx_triad(mtx1, mtx2):
    # R*R, R*R^T, R^T*R, R^T*R^T
    mtx = [mtx1 * mtx2,
           mtx1 * csr_matrix.transpose(mtx2),
           (csr_matrix.transpose(mtx1) * mtx2).tocsr(),
           (csr_matrix.transpose(mtx1) * csr_matrix.transpose(mtx2)).tocsr()]
    return mtx


# create matrix for feature extraction
def create_mtx(df, df_pos, df_neg, n_user):
    # pos -> neg
    # neg -> pos 
    # 값은 모두 1
    temp1 = csr_matrix((df.feedback, (df.sid, df.tid)), shape=(n_user, n_user))
    temp2 = csr_matrix((df.feedback, (df.tid, df.sid)), shape=(n_user, n_user))
    # user-item, item-user 모두 연결 된 행렬
    mtx_train_csr = temp1 + temp2
    mtx_train_csr.data = np.ones_like(mtx_train_csr.data)
    # 곱해서 서로의 연결을 미리 가중하는것인가?
    mtx_common = mtx_train_csr * mtx_train_csr

    mtx_cs = [df_to_csr_matrix(df_pos, n_user), df_to_csr_matrix(df_neg, n_user),
              df_to_csc_matrix(df_pos, n_user), df_to_csc_matrix(df_neg, n_user)]

    mtx_triad = []
    mtx_triad.extend(set_mtx_triad(mtx_cs[0], mtx_cs[0])) # pos, pos
    mtx_triad.extend(set_mtx_triad(mtx_cs[0], mtx_cs[1])) # pos, neg
    mtx_triad.extend(set_mtx_triad(mtx_cs[1], mtx_cs[0])) # neg, pos
    mtx_triad.extend(set_mtx_triad(mtx_cs[1], mtx_cs[1])) # neg, neg

    mtx = [mtx_cs, mtx_common, mtx_triad]
    return mtx


# set matrix for TrustSGCN
def set_TrustSGCN_mtx( scores, signs, n_user, sign_thres, adj_dic_all):
    cnt = 0
    
    # Trust, sign=1
    lists_T1 = list()
    # Trust, sign=-1
    lists_T2 = list()
    # Untrust, sign=1
    lists_U1 = list()
    # Untrust, sign=-1
    lists_U2 = list()

    pbar=tqdm(total=len(adj_dic_all.keys()))
    for i in adj_dic_all.keys():
        for j, sign, hop in adj_dic_all.get(i):
            cnt +=  1
            fifi = [i, j, sign, hop]
            #1hop은 다 믿어
            if hop == 1 and sign == 1:
                # Trust의 약자구나. Trust hop1
                lists_T1.append(fifi)

            elif hop == 1 and sign == -1:
                lists_T2.append(fifi)
      
            # mesure trustwothiness based two condition
            # hop이 2이상일 때 검사
            elif(sign == signs[i][j]): # condition 1: sign predicted by balance theory ==  sign predicted by fextra ?
                if sign==1: # pos edge
                    if(scores[i][j]>sign_thres[0]): # condition 2: over threshold?
                        lists_T1.append(fifi)
                    else:
                        lists_U1.append(fifi)
                        
                if sign==-1:
                    if(abs(scores[i][j])>sign_thres[1]): # condition 2: over threshold?
                        lists_T2.append(fifi)
       
                    else: 
                        lists_U2.append(fifi)
                    
                    
            elif((hop != 1) and (sign != signs[i][j])):
                if sign == 1:
                    lists_U1.append(fifi)
                        
                if sign==-1:
                    lists_U2.append(fifi)

        pbar.update(1)
    pbar.close()  

    return lists_T1, lists_T2, lists_U1, lists_U2

# Only Trustworthy GCN
def set_Balance_mtx( scores, signs, n_user, sign_thres, adj_dic_all):
    for i in range(n_user):
        for j in range(n_user):
            if(signs[i][j]==0):
                signs[i][j]=-1
    
    lists_T1 = list()
    lists_T2 = list()
    lists_U1 = list()
    lists_U2 = list()

    pbar=tqdm(total=len(adj_dic_all.keys()))
    for i in adj_dic_all.keys():
        for j, sign, hop in adj_dic_all.get(i):
            fifi = [i, j, sign, hop]
            if sign == 1:
                lists_T1.append(fifi)

            elif sign == -1:
                lists_T2.append(fifi)
        pbar.update(1)
    pbar.close()  

    return lists_T1, lists_T2, lists_U1, lists_U2 

# Only FExtra GCN
def set_FExtra_mtx( scores, signs, n_user, sign_thres, adj_dic_all):
 
    lists_T1 = list()
    lists_T2 = list()
    lists_U1 = list()
    lists_U2 = list()

    pbar=tqdm(total=len(adj_dic_all.keys()))
    for i in adj_dic_all.keys():
        for j, sign, hop in adj_dic_all.get(i):
            fifi = [i, j, sign, hop]
            if signs[i][j] == 1 :
                lists_T1.append(fifi)

            elif signs[i][j] == -1:
                lists_T2.append(fifi)
        pbar.update(1)
    pbar.close()  

    return lists_T1, lists_T2, lists_U1, lists_U2


def get_degree(mtx, idx):
    # 해당 행에 존재하는 data의 개수만 반환
    return mtx.indptr[idx + 1] - mtx.indptr[idx]
