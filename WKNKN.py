import numpy as np
import pandas as pd


def WKNKN(Y, Sd, St, K1,K2):
    """
    WKNKN算法，用于填补RNA-疾病关联矩阵中的缺失值

    参数:
    Y  - RNA-疾病关联矩阵 (m x n)，其中 m 是 RNA 的数量，n 是疾病的数量
    Sd - 疾病相似性矩阵 (n x n)
    St - RNA相似性矩阵 (m x m)
    K  - 最近邻的数量

    返回:
    Y  - 填补缺失值后的 RNA-疾病关联矩阵
    """

    # 初始化两个矩阵 Yd 和 Yt，用于存储疾病和RNA方向的加权结果
    Yd = np.zeros_like(Y)  # 疾病方向的加权结果
    Yt = np.zeros_like(Y)  # RNA方向的加权结果

    # 疾病方向：遍历每个疾病列
    for d in range(Y.shape[1]):
        # 获取疾病 d 的 K 个最近邻
        dnn = KNearestKnownNeighbors(Sd[d, :], K1)

        # 初始化权重和归一化项
        weights = np.zeros(K1)
        normalization_term = 0

        # 计算每个邻居的权重
        for i in range(K1):
            weights[i] = (1 / (i + 1)) * Sd[d, dnn[i]]
            normalization_term += Sd[d, dnn[i]]

        # 根据权重计算 Yd
        for i in range(K1):
            Yd[:, d] += (weights[i] / normalization_term) * Y[:, dnn[i]]

    # RNA方向：遍历每个RNA行
    for t in range(Y.shape[0]):
        # 获取RNA t 的 K 个最近邻
        tnn = KNearestKnownNeighbors(St[t, :], K1)

        # 初始化权重和归一化项
        weights = np.zeros(K1)
        normalization_term = 0

        # 计算每个邻居的权重
        for j in range(K1):
            weights[j] = (1 / (j + 1)) * St[t, tnn[j]]
            normalization_term += St[t, tnn[j]]

        # 根据权重计算 Yt
        for j in range(K1):
            Yt[t, :] += (weights[j] / normalization_term) * Y[tnn[j], :]

    # 将 Yd 和 Yt 的加权平均合并
    Ydt = (Yd + Yt) / 2

    # 最终输出矩阵：取 Y 和 Ydt 中的每个值的最大值
    Y = np.maximum(Y, Ydt)

    return Y

def KNearestKnownNeighbors(similarity_row, K):
    """
    获取给定相似度行中的前 K 个最大相似度的索引

    参数:
    similarity_row - 相似性矩阵的一行
    K              - 最近邻数量

    返回:
    neighbors      - 前 K 个最大相似度的邻居索引
    """
    # 按相似度从大到小排序并返回对应索引
    neighbors = np.argsort(similarity_row)[::-1][:K]
    return neighbors
association = pd.read_csv("../adj_index.csv", index_col=0).to_numpy()
# sim_m1 = pd.read_csv('../sno_p2p_smith.csv', index_col=0).to_numpy()
# sim_m2=np.load('./GIPK-s.npy')
# ss=(sim_m1+sim_m2)/2
ss = pd.read_csv('./s_fusion.csv', index_col=0).to_numpy()
# sim_d1 = pd.read_csv('../disease_similarity.csv', index_col=0).to_numpy()
# sim_d2=np.load('./GIPK-d.npy')
# ds=(sim_d1+sim_d2)/2
ds = pd.read_csv('./d_fusion.csv', index_col=0).to_numpy()
k1=15
k2=5
A=WKNKN(association,ds,ss,k1,k2)
print(A.shape)
print(A)
pd.DataFrame(A).to_csv('G:/Python/ED_SDA/NMF/fusion_WKNKN-15.csv',index=True)