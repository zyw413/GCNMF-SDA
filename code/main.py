import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import KFold, StratifiedKFold
from scipy import interp
import numpy as np
import warnings
from utils import *

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# === EarlyStopping by AUC ===
class EarlyStoppingAUC:
    def __init__(self, patience=200, verbose=False, delta=0.0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_auc = 0.0
        self.delta = delta
        self.path = path

    def __call__(self, val_auc, model):
        score = val_auc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_auc, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # if self.verbose:
            #     print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_auc, model)
            self.counter = 0

    def save_checkpoint(self, val_auc, model):
        # if self.verbose:
        #     print(f"Validation AUC improved. Saving model ...")
        torch.save(model.state_dict(), self.path)
        self.best_auc = val_auc

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, hidden_dim1,out_dim, dropout):
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


def samples_choose(rel_adj_mat, features_embedding_rna, features_embedding_dis, negative_sample_times=1, seed=42):
    # 设置随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)

    X, y = [], []

    # 转为 Tensor
    rel_adj_mat = torch.tensor(rel_adj_mat.values)
    features_embedding_rna = torch.tensor(features_embedding_rna.values, dtype=torch.float32)
    features_embedding_dis = torch.tensor(features_embedding_dis.values, dtype=torch.float32)

    # 正样本索引
    positive_index_tuple = torch.where(rel_adj_mat == 1)
    positive_index_list = list(zip(positive_index_tuple[0].tolist(), positive_index_tuple[1].tolist()))

    for (r, d) in positive_index_list:
        # 正样本
        pos_sample = torch.cat([features_embedding_rna[r], features_embedding_dis[d]], dim=0).unsqueeze(0)
        X.append(pos_sample)
        y.append(1)

        # 负样本
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
n_splits = 5
epochs = 2000
fold = 0
result = np.zeros((1, 7), float)
K=5
D = 45
adj = pd.read_csv(r'G:/Python/GCNMF-SDA/data/adj_index.csv',index_col=0,header=0)
ss = pd.read_csv('G:/Python/GCNMF-SDA/data/s_fusion.csv', index_col=0).to_numpy()
ds = pd.read_csv('G:/Python/GCNMF-SDA/data/d_fusion.csv', index_col=0).to_numpy()
association = pd.read_csv("G:/Python/GCNMF-SDA/data/adj_index.csv", index_col=0).to_numpy()
A_WKNKN=WKNKN(association,ds,ss,K)
A_WKNKN_df = pd.DataFrame(A_WKNKN).to_numpy()
nmf_s, nmf_d = get_low_feature(D, 0.01, pow(10, -4), A_WKNKN)
ass_path = 'G:/Python/GCNMF-SDA/data/ass.txt'
ss=pd.DataFrame(ss)
ds=pd.DataFrame(ds)
gcn_s,gcn_d=extract_gcn_features(ass_path, ss, ds,A_WKNKN)
s_feature = pd.concat([nmf_s, gcn_s], axis=1)
d_feature = pd.concat([nmf_d, gcn_d], axis=1)
X,y=samples_choose(adj,s_feature,d_feature)
kf =StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
for train_index, val_index in kf.split(X, y):
    fold += 1
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    train_feature = torch.tensor(X_train, dtype=torch.float32).to(device)
    train_label = torch.tensor(y_train, dtype=torch.float32).to(device)
    val_feature = torch.tensor(X_val, dtype=torch.float32).to(device)
    val_label = torch.tensor(y_val, dtype=torch.float32).to(device)

    input_dim = train_feature.shape[1]
    dropout = 0
    model = MLP(in_dim=input_dim, hidden_dim=128, hidden_dim1=64, out_dim=1, dropout=dropout).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    early_stopping = EarlyStoppingAUC(patience=40, verbose=True, path=f'checkpoint.pt')



    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(train_feature).squeeze()
        loss = criterion(outputs, train_label)
        loss.backward()
        optimizer.step()

        with torch.no_grad():

            val_outputs = model(val_feature).squeeze()
            val_loss = criterion(val_outputs, val_label)
            y_score = val_outputs.cpu().numpy()
            val_auc = auc(*roc_curve(val_label.cpu().numpy(), y_score)[:2])
        if epoch >400:  # after minimum epoch, check early stopping
            early_stopping(val_auc, model)
            if early_stopping.early_stop:
                    print(f"Early stopping at epoch {epoch + 1} with best AUC: {early_stopping.best_auc:.4f}")
                    break

    # model.load_state_dict(torch.load(f'best_model_fold{fold}.pt'))
    # model.eval()

    with torch.no_grad():
        y_score = model(val_feature).cpu().numpy()
    metrics = get_metrics(val_label.cpu().numpy(), y_score)
    result += metrics
    print('[auc,aupr,  f1_score, accuracy, recall, specificity, precision]', metrics)
print("==================================================")
print(result / n_splits)