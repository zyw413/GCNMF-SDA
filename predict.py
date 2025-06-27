import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from utils import *
import os

# 复制samples_choose函数

def samples_choose(rel_adj_mat, features_embedding_rna, features_embedding_dis, negative_sample_times=1, seed=42):
    X, y = [], []
    rel_adj_mat = torch.tensor(rel_adj_mat.values)
    features_embedding_rna = torch.tensor(features_embedding_rna.values, dtype=torch.float32)
    features_embedding_dis = torch.tensor(features_embedding_dis.values, dtype=torch.float32)
    positive_index_tuple = torch.where(rel_adj_mat == 1)
    positive_index_list = list(zip(positive_index_tuple[0].tolist(), positive_index_tuple[1].tolist()))
    for (r, d) in positive_index_list:
        pos_sample = torch.cat([features_embedding_rna[r], features_embedding_dis[d]], dim=0).unsqueeze(0)
        X.append(pos_sample)
        y.append(1)
        negative_colindex_list = []
        for _ in range(negative_sample_times):
            j = np.random.randint(rel_adj_mat.size(1))
            while (r, j) in positive_index_list:
                j = np.random.randint(rel_adj_mat.size(1))
            negative_colindex_list.append(j)
        for j in negative_colindex_list:
            neg_sample = torch.cat([features_embedding_rna[r], features_embedding_dis[j]], dim=0).unsqueeze(0)
            X.append(neg_sample)
            y.append(0)
    X = torch.cat(X, dim=0)
    y = torch.FloatTensor(y)
    return X, y

# 设备
seed = 2021
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义MLP，与main.py保持一致
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, hidden_dim1, out_dim, dropout):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim1)
        self.fc3 = nn.Linear(hidden_dim1, out_dim)
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        x = self.relu(self.fc2(x))
        x = self.drop(x)
        x = self.sigmoid(self.fc3(x))
        return x

class GCNMF_SDA:
    def __init__(self, device=None):
        self.K = 5
        self.D = 45
        self.device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = None
        self.input_dim = None
        self.dropout = 0
        self.X = None
        self.y = None

    def prepare_features(self):
        adj = pd.read_csv(r'G:/Python/GCNMF-SDA/data/adj_index.csv', index_col=0, header=0)
        ss = pd.read_csv('G:/Python/GCNMF-SDA/data/s_fusion.csv', index_col=0).to_numpy()
        ds = pd.read_csv('G:/Python/GCNMF-SDA/data/d_fusion.csv', index_col=0).to_numpy()
        association = pd.read_csv("G:/Python/GCNMF-SDA/data/adj_index.csv", index_col=0).to_numpy()
        A_WKNKN = WKNKN(association, ds, ss, self.K)
        nmf_s, nmf_d = get_low_feature(self.D, 0.01, pow(10, -4), A_WKNKN)
        ass_path = 'G:/Python/GCNMF-SDA/data/ass.txt'
        ss = pd.DataFrame(ss)
        ds = pd.DataFrame(ds)
        gcn_s, gcn_d = extract_gcn_features(ass_path, ss, ds, A_WKNKN)
        s_feature = pd.concat([nmf_s, gcn_s], axis=1)
        d_feature = pd.concat([nmf_d, gcn_d], axis=1)
        X, y = samples_choose(adj, s_feature, d_feature)
        self.X = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.y = torch.tensor(y, dtype=torch.float32).to(self.device)
        self.input_dim = self.X.shape[1]

    def build_model(self):
        self.model = MLP(in_dim=self.input_dim, hidden_dim=128, hidden_dim1=64, out_dim=1, dropout=self.dropout).to(self.device)

    def load_weights(self, weight_path='checkpoint.pt'):
        if self.model is None:
            self.build_model()
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        self.model.eval()

    def predict(self):
        if self.X is None or self.model is None:
            raise RuntimeError('Features or model not prepared!')
        with torch.no_grad():
            y_pred = self.model(self.X).cpu().numpy().flatten()
        return y_pred, self.y.cpu().numpy()

if __name__ == '__main__':
    seed = 2021
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    gcnmf_sda = GCNMF_SDA()
    gcnmf_sda.prepare_features()
    gcnmf_sda.build_model()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    weight_path = os.path.join(base_dir, 'checkpoint.pt')
    gcnmf_sda.load_weights(weight_path)
    y_pred, y_true = gcnmf_sda.predict()
    np.savetxt('predict_scores.txt', y_pred)
    np.savetxt('predict_labels.txt', y_true)
    print('预测完成，结果已保存到 predict_scores.txt 和 predict_labels.txt') 