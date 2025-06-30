import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, auc, roc_curve
warnings.filterwarnings("ignore")

# === EarlyStopping by AUC
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
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_auc, model)
            self.counter = 0

    def save_checkpoint(self, val_auc, model):
        if self.verbose:
            print(f"Validation AUC improved. Saving model ...")
        torch.save(model.state_dict(), self.path)
        self.best_auc = val_auc

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, hidden_dim1, out_dim, dropout=0):
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


def samples_choose(rel_adj_mat, features_embedding_rna, features_embedding_dis,disease_exclude_index,negative_sample_times=1, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

    X, y = [], []
    # adj_mat_tensor= torch.tensor(adj_mat.values)
    rel_adj_mat = torch.tensor(rel_adj_mat.values)
    features_embedding_rna = torch.tensor(features_embedding_rna.values, dtype=torch.float32)
    features_embedding_dis = torch.tensor(features_embedding_dis.values, dtype=torch.float32)

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
            while (r, j) in positive_index_list or(j == disease_exclude_index):
                j = np.random.randint(rel_adj_mat.size(1))
            negative_colindex_list.append(j)

        for j in negative_colindex_list:
            neg_sample = torch.cat([features_embedding_rna[r], features_embedding_dis[j]], dim=0).unsqueeze(0)
            X.append(neg_sample)
            y.append(0)

    X = torch.cat(X, dim=0)
    y = torch.FloatTensor(y)
    return X, y

gcn_s_features = np.load('G:/Python/ED_SDA/NMF/gmodel_feat_sno_fusion-2-WKNKN.npy')
gcn_d_features = np.load('G:/Python/ED_SDA/NMF/gmodel_feat_dis_fusion-2-WKNKN.npy')
nmf_s = pd.read_csv('G:/Python/ED_SDA/NMF/NMF_sfeature.csv', index_col=0)
nmf_d = pd.read_csv('G:/Python/ED_SDA/NMF/NMF_dfeature.csv', index_col=0)

features_embedding_rna = pd.concat([nmf_s, pd.DataFrame(gcn_s_features)], axis=1)
features_embedding_dis = pd.concat([nmf_d, pd.DataFrame(gcn_d_features)], axis=1)
adj = pd.read_csv(r'G:/Python/ED_SDA/adj_index.csv', index_col=0, header=0)

# === 将 adj 的索引赋给 features_embedding_rna 和 features_embedding_dis ===
features_embedding_rna.index = adj.index
features_embedding_dis.index = adj.columns
# === 2. 设置目标疾病 ===
disease_id = 'Lung Cancer'


rel_adj_mat = adj.copy()
if disease_id in rel_adj_mat.columns:
    rel_adj_mat[disease_id] = 0
disease_exclude_index = adj.columns.get_loc(disease_id)
X, y = samples_choose(
    rel_adj_mat=rel_adj_mat,
    features_embedding_rna=features_embedding_rna,
    features_embedding_dis=features_embedding_dis,
    negative_sample_times=1,
    seed=22,
    disease_exclude_index=disease_exclude_index,
)

# === 4. 划分训练验证集 ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2,stratify=y,random_state=42)
# === 5. 初始化模型/损失函数/优化器 ===
input_dim = X.shape[1]
model = MLP(in_dim=input_dim, hidden_dim=128, hidden_dim1=64, out_dim=1)
criterion = nn.BCELoss()
# optimizer = optim.RMSprop(model.parameters(), lr=0.01)
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# === 6. 初始化 EarlyStopping ===
early_stopping = EarlyStoppingAUC(patience=20, verbose=True)

# === 7. 训练模型 ===
epochs = 2000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train).squeeze()
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val).squeeze().cpu().numpy()
        val_loss = criterion(torch.tensor(val_outputs), torch.tensor(y_val))
        # val_auc = auc(*roc_curve(y_val.cpu().numpy(), val_outputs)[:2])
        val_auc,aupr,  f1_score, accuracy, recall, specificity, precision = get_metrics(y_val.cpu().numpy(),val_outputs)

    print(f"Epoch {epoch+1}, Train Loss: {loss.item():.4f}, Val AUC: {val_auc:.4f}")

    if epoch > 600:
        early_stopping(val_auc, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break
model.load_state_dict(torch.load('best_model_fold2.pt'))
# model.eval()
# === 8. 预测目标疾病与所有 snoRNAs 的关联 ===
with torch.no_grad():
    if disease_id not in features_embedding_dis.index:
        raise ValueError(f"Error: Disease ID '{disease_id}' not found in features_embedding_dis!")

    dis_embed = features_embedding_dis.loc[disease_id].values
    X_disease = []
    rna_names = []

    for rna in features_embedding_rna.index:
        rna_embed = features_embedding_rna.loc[rna].values
        combined = np.concatenate([rna_embed, dis_embed])
        X_disease.append(combined)
        rna_names.append(rna)

    X_disease = torch.tensor(np.array(X_disease), dtype=torch.float32)
    predictions = model(X_disease).cpu().numpy()

# === 9. 保存预测结果 ===
results = pd.DataFrame({
    'snoRNA-disease pair': [f"{rna} - {disease_id}" for rna in rna_names],
    'Predicted Score': predictions.flatten()
})

results = results.sort_values(by='Predicted Score', ascending=False).head(307)
results.to_csv(f'predicted_associations_for_{disease_id}_after_deletion3.csv', index=False)

print(f"Predictions for '{disease_id}' saved to 'predicted_associations_for_{disease_id}_after_deletion3.csv'.")

# # === 8. 预测目标 snoRNA 与所有疾病的关联 ===
# with torch.no_grad():
#     if snoRNA_id not in features_embedding_rna.index:
#         raise ValueError(f"Error: snoRNA ID '{snoRNA_id}' not found!")
#
#     rna_embed = features_embedding_rna.loc[snoRNA_id].values
#     X_rna = []
#     disease_names = []
#
#     for disease in features_embedding_dis.index:
#         dis_embed = features_embedding_dis.loc[disease].values
#         combined = np.concatenate([rna_embed, dis_embed])
#         X_rna.append(combined)
#         disease_names.append(disease)
#
#     X_rna = torch.tensor(np.array(X_rna), dtype=torch.float32)
#     predictions = model(X_rna).cpu().numpy()
#
# # === 9. 保存结果 ===
# results = pd.DataFrame({
#     'snoRNA-disease pair': [f"{snoRNA_id} - {d}" for d in disease_names],
#     'Predicted Score': predictions.flatten()
# })
# results = results.sort_values(by='Predicted Score', ascending=False).head(54)
# results.to_csv(f'predicted_associations_for_{snoRNA_id}_after_deletion.csv', index=False)
#
# print(f"Predictions for '{snoRNA_id}' saved to 'predicted_associations_for_{snoRNA_id}_after_deletion.csv'.")
