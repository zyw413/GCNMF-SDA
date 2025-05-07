import dgl
import torch
import dgl.nn.pytorch as dglnn
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def WKNKN(Y, Sd, St, K):
    # 初始化两个矩阵 Yd 和 Yt，用于存储疾病和RNA方向的加权结果
    Yd = np.zeros_like(Y)  # 疾病方向的加权结果
    Yt = np.zeros_like(Y)  # RNA方向的加权结果

    # 疾病方向：遍历每个疾病列
    for d in range(Y.shape[1]):
        # 获取疾病 d 的 K 个最近邻
        dnn = KNearestKnownNeighbors(Sd[d, :], K)

        # 初始化权重和归一化项
        weights = np.zeros(K)
        normalization_term = 0

        # 计算每个邻居的权重
        for i in range(K):
            weights[i] = (1 / (i + 1)) * Sd[d, dnn[i]]
            normalization_term += Sd[d, dnn[i]]

        # 根据权重计算 Yd
        for i in range(K):
            Yd[:, d] += (weights[i] / normalization_term) * Y[:, dnn[i]]

    # RNA方向：遍历每个RNA行
    for t in range(Y.shape[0]):
        # 获取RNA t 的 K 个最近邻
        tnn = KNearestKnownNeighbors(St[t, :], K)

        # 初始化权重和归一化项
        weights = np.zeros(K)
        normalization_term = 0

        # 计算每个邻居的权重
        for j in range(K):
            weights[j] = (1 / (j + 1)) * St[t, tnn[j]]
            normalization_term += St[t, tnn[j]]

        # 根据权重计算 Yt
        for j in range(K):
            Yt[t, :] += (weights[j] / normalization_term) * Y[tnn[j], :]

    # 将 Yd 和 Yt 的加权平均合并
    Ydt = (Yd + Yt) / 2

    # 最终输出矩阵：取 Y 和 Ydt 中的每个值的最大值
    Y = np.maximum(Y, Ydt)

    return Y

def KNearestKnownNeighbors(similarity_row, K):
    # 按相似度从大到小排序并返回对应索引
    neighbors = np.argsort(similarity_row)[::-1][:K]
    return neighbors

def get_low_feature(k,lam, th, A):
    m, n = A.shape
    arr1=np.random.randint(0,100,size=(m,k))
    U = arr1/100
    arr2=np.random.randint(0,100,size=(k,n))
    V = arr2/100
    obj_value = objective_function(A, A, U, V, lam)
    obj_value1 = obj_value + 1
    i = 0
    diff = abs(obj_value1 - obj_value)
    while i < 1000:
        i =i + 1
        U = updating_U(A, A, U, V, lam)
        V = updating_V(A, A, U, V, lam)

    return pd.DataFrame(U), pd.DataFrame(V.transpose())

def objective_function(W, A, U, V, lam):
    m, n = A.shape
    sum_obj = 0
    for i in range(m):
        for j in range(n):
            sum_obj = sum_obj + W[i,j]*(A[i,j] - U[i,:].dot(V[:,j]))+ lam*(np.linalg.norm(U[i, :], ord=1,keepdims= False) + np.linalg.norm(V[:, j], ord = 1, keepdims = False))
    return  sum_obj

def updating_U (W, A, U, V, lam):
    m, n = U.shape
    upper = (W*A).dot(V.T)
    down = (W*(U.dot(V))).dot((V.T)) + (lam/2) *(np.ones([m, n]))
    U_new = U
    for i in range(m):
        for j in range(n):
            U_new[i,j] = U[i, j]*(upper[i,j]/down[i, j])
    return U_new


def updating_V (W, A, U, V, lam):
        m,n = V.shape
        upper = (U.T).dot(W*A)
        down = (U.T).dot(W*(U.dot(V)))+(lam/2)*(np.ones([m,n]))
        V_new = V
        for i in range(m):
            for j in range(n):
                V_new[i,j] = V[i, j]*(upper[i,j]/down[i,j])
        return V_new

def get_ass(file_path):
    snoRNA_name = []
    disease_name = []
    snoRNA_disease = []

    f = open(file_path, 'r', encoding='utf-8')
    contents = f.readlines()
    for content in contents:
        value = content.strip().split(',')
        value[0] = value[0].lower()
        if value[0] not in snoRNA_name: snoRNA_name.append(value[0])
        value[1] = value[1].lower().strip('\n')
        if value[1] not in disease_name: disease_name.append(value[1])
        snoRNA_disease.append(value)
    f.close()
    snoRNA_num = len(snoRNA_name)
    disease_num = len(disease_name)

    # print(snoRNA_num)
    # print(disease_num)

    snoRNA_index = dict(zip(snoRNA_name, range(0, snoRNA_num)))
    disease_index = dict(zip(disease_name, range(0, disease_num)))

    input_snoRNA_disease = [[], []]
    for i in range(len(snoRNA_disease)):
        input_snoRNA_disease[0].append(snoRNA_index.get(snoRNA_disease[i][0]))
        input_snoRNA_disease[1].append(disease_index.get(snoRNA_disease[i][1]))

    # print(len(input_snoRNA_disease[0]))

    return input_snoRNA_disease
def construct_initial_feature_matrix(SS, SD, Y):
    """
    构建异质图的初始特征矩阵 ISD
    ISD = [ IS ; ID ] = [ [SS, Y]; [Y_T, SD] ]
    """
    # 计算 Y 的转置矩阵
    Y_T = Y.T

    # 构建 snoRNA 同质图 GS 的初始特征矩阵 IS
    IS = np.concatenate((SS, Y), axis=1)

    # 构建疾病同质图 GD 的初始特征矩阵 ID
    ID = np.concatenate((Y_T, SD), axis=1)

    return pd.DataFrame(IS),pd.DataFrame(ID)
def Graph_create(snoRNA_disease, snoRNA_feat, disease_feat):
    graph = {
        ('snoRNA', 's_d', 'disease'): (torch.tensor(snoRNA_disease[0]), torch.tensor(snoRNA_disease[1])),
        ('disease', 'd_s', 'snoRNA'): (torch.tensor(snoRNA_disease[1]), torch.tensor(snoRNA_disease[0])),
    }

    graph = dgl.heterograph(graph)

    graph.nodes['snoRNA'].data['feature'] = snoRNA_feat
    graph.nodes['disease'].data['feature'] = disease_feat

    graph_h = {'snoRNA': graph.nodes['snoRNA'].data['feature'],
               'disease': graph.nodes['disease'].data['feature']}

    return graph, graph_h
class GCN(nn.Module):
    def __init__(self, C_dim, S_dim, hidden_dim,out_dim):
        super(GCN, self).__init__()

        self.C_dim = C_dim
        self.S_dim = S_dim

        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.HeteroConv1 = dglnn.HeteroGraphConv({
            's_d': dglnn.GraphConv(self.C_dim, self.hidden_dim, activation=F.relu, allow_zero_in_degree=True),
            'd_s': dglnn.GraphConv(self.S_dim, self.hidden_dim, activation=F.relu, allow_zero_in_degree=True)
        },
            aggregate='sum')

        # self.HeteroConv2 = dglnn.HeteroGraphConv({
        #     's_d': dglnn.GraphConv(self.hidden_dim, self.hidden_dim, activation=F.relu, allow_zero_in_degree=True),
        #     'd_s': dglnn.GraphConv(self.hidden_dim, self.hidden_dim, activation=F.relu, allow_zero_in_degree=True)
        # },
        #     aggregate='sum')

        self.HeteroConv3 = dglnn.HeteroGraphConv({
            's_d': dglnn.GraphConv(self.hidden_dim, self.out_dim, activation=F.relu, allow_zero_in_degree=True),
            'd_s': dglnn.GraphConv(self.hidden_dim, self.out_dim, activation=F.relu, allow_zero_in_degree=True)
        },
            aggregate='sum')

    def forward(self, g, h):

        h1 = self.HeteroConv1(g, h)
        # h2 = self.HeteroConv2(g, h1)
        h3 = self.HeteroConv3(g, h1)

        return h3


def extract_gcn_features(ass_path, ss, ds,A_WKNKN):

    # Load association data
    snoRNA_disease = get_ass(ass_path)

    # Construct initial feature matrices
    snoRNA_feat, dis_feat = construct_initial_feature_matrix(ss, ds, A_WKNKN)
    snoRNA_feat = torch.from_numpy(np.array(snoRNA_feat, dtype='float32'))
    dis_feat = torch.from_numpy(np.array(dis_feat, dtype='float32'))

    # Create graph structures
    graph, graph_h = Graph_create(snoRNA_disease, snoRNA_feat, dis_feat)

    # Initialize and run the GCN model
    model = GCN(C_dim=361, S_dim=361, hidden_dim=128,out_dim=64)
    h = model(graph, graph_h)

    # Extract and convert features to DataFrames
    gcn_s = pd.DataFrame(h['snoRNA'].detach().numpy())
    gcn_d = pd.DataFrame(h['disease'].detach().numpy())

    return gcn_s, gcn_d
def get_metrics(real_score, predict_score):
    predict_score=predict_score.flatten()
    sorted_predict_score = np.array(
        sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(
        sorted_predict_score_num*np.arange(1, 1000)/1000)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    # print(predict_score_matrix.shape)
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)

    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T) #
    FP = predict_score_matrix.sum(axis=1)-TP
    FN = real_score.sum()-TP
    TN = len(real_score.T)-TP-FP-FN

    fpr = FP/(FP+TN)
    tpr = TP/(TP+FN)
    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T
    auc = 0.5*(x_ROC[1:]-x_ROC[:-1]).T*(y_ROC[:-1]+y_ROC[1:])

    recall_list = tpr
    precision_list = TP/(TP+FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack(
        (recall_list, precision_list)).tolist())).T
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5*(x_PR[1:]-x_PR[:-1]).T*(y_PR[:-1]+y_PR[1:])

    f1_score_list = 2*TP/(len(real_score.T)+TP-TN)
    accuracy_list = (TP+TN)/len(real_score.T)
    specificity_list = TN/(TN+FP)

    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]
    return [auc[0, 0],aupr[0, 0], f1_score, accuracy, recall, specificity, precision]

def get_fusion_sim (k1, k2):

    sim_m1=pd.read_csv('./sno_p2p_smith.csv', index_col=0).to_numpy()
    sim_m2=np.load('./GIPK-s.npy')

    sim_d1=pd.read_csv('./disease_similarity.csv', index_col=0).to_numpy()
    sim_d2=np.load('./GIPK-d.npy')
    m1 = new_normalization1(sim_m1)
    m2 = new_normalization1(sim_m2)
    # m3 = new_normalization1(sim_m3)

    Sm_1 = KNN_kernel1(sim_m1, k1)
    Sm_2 = KNN_kernel1(sim_m2, k1)
    # Sm_3 = KNN_kernel1(sim_m3, k1)

    Pm = Updating1(Sm_1,Sm_2, m1, m2)
    Pm_final = (Pm + Pm.T)/2


    d1 = new_normalization1(sim_d1)
    d2 = new_normalization1(sim_d2)
    # d3 = new_normalization1(sim_d3)


    Sd_1 = KNN_kernel1(sim_d1, k2)
    Sd_2 = KNN_kernel1(sim_d2, k2)
    # Sd_3 = KNN_kernel1(sim_d3, k2)

    Pd = Updating1(Sd_1,Sd_2, d1, d2)
    Pd_final = (Pd+Pd.T)/2



    return Pm_final, Pd_final


def new_normalization1 (w):
    m = w.shape[0]
    p = np.zeros([m,m])
    for i in range(m):
        for j in range(m):
            if i == j:
                p[i][j] = 1/2
            elif np.sum(w[i,:])-w[i,i]>0:
                p[i][j] = w[i,j]/(2*(np.sum(w[i,:])-w[i,i]))
    return p


def KNN_kernel1 (S, k):
    n = S.shape[0]
    S_knn = np.zeros([n,n])
    for i in range(n):
        sort_index = np.argsort(S[i,:])
        for j in sort_index[n-k:n]:
            if np.sum(S[i,sort_index[n-k:n]])>0:
                S_knn [i][j] = S[i][j] / (np.sum(S[i,sort_index[n-k:n]]))
    return S_knn



def Updating1 (S1,S2, P1,P2):
    it = 0
    P = (P1+P2)/2
    dif = 1
    while dif>0.0000001:
        it = it + 1
        P111 =np.dot (np.dot(S1,P2),S1.T)
        P111 = new_normalization1(P111)
        P222 =np.dot (np.dot(S2,P1),S2.T)
        P222 = new_normalization1(P222)
        # P333 = np.dot (np.dot(S3,(P1+P2)/2),S3.T)
        # P333 = new_normalization1(P333)

        P1 = P111
        P2 = P222
        # P3 = P333

        P_New = (P1+P2)/2
        dif = np.linalg.norm(P_New-P)/np.linalg.norm(P)
        P = P_New
    print("Iter numb1", it)
    return P
