import itertools
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import  roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import KFold
from scipy import interp
from classifiers import *

from NMF import get_low_feature, generate_f1
# from GAE_trainer import *
# from GAE import *
from metric import *
# from similarity_fusion import *
# from five_AE import *
import warnings
# from model import MLP

warnings.filterwarnings("ignore")
seed = 516
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
# 参数
n_splits = 5
classifier_epochs = 2000
m_threshold = [0.5]
epochs = [300]
fold = 0
result = np.zeros((1, 7), float)
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 1000)
precisions = []
auprs = []
mean_recall = np.linspace(0, 1, 1000)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fold_losses = []
val_fold_losses=[]
# 定义 MLP 模型类
class MLPClassifier(nn.Module):
    def __init__(self, input_dim):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        # x=F.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = torch.relu(self.fc2(x))
        # x = F.leaky_relu(self.fc2(x), negative_slope=0.01)
        x = self.sigmoid(self.fc3(x))
        return x


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, hidden_dim1,out_dim, dropout):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim1)
        self.fc3 = nn.Linear(hidden_dim1, out_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        x = self.relu(self.fc2(x))
        x = self.drop(x)
        x = self.sigmoid(self.fc3(x))
        return x
# 定义保存预测值和对应的非编码RNA和疾病对的函数
def save_predictions_and_pairs(predictions, pairs, fold):
    predictions = predictions.cpu().numpy()  # 确保predictions在CPU上并转换为numpy数组
    pairs = pairs.cpu().numpy() if isinstance(pairs, torch.Tensor) else pairs  # 确保pairs是numpy数组
    np.savetxt(f'val_outputs_fold_{fold}.txt', predictions)  # 保存预测值
    np.savetxt(f'val_pairs_fold_{fold}.txt', pairs)  # 保存非编码RNA和疾病对的索引
for s in itertools.product(m_threshold, epochs):
    association = pd.read_csv("../adj_index.csv", index_col=0).to_numpy()
    # associations = pd.read_csv("../Edge supplementation/ED_adj.csv", index_col=0).to_numpy()
    # associations = pd.read_csv("../NMF/WKNKN-5-30.csv", index_col=0).to_numpy()
    associations = pd.read_csv("../NMF/fusion_WKNKN-5.csv", index_col=0).to_numpy()
    samples = get_all_samples(association)

    # k1 = 30
    # k2 =5
    # m_fusion_sim, d_fusion_sim = get_fusion_sim(k1, k2)

    # ss = pd.read_csv('../sno_p2p_smith.csv', index_col=0)
    # ds = pd.read_csv('../disease_similarity.csv', index_col=0)
    kf = KFold(n_splits=n_splits, shuffle=True,random_state=42)

    D = 45
    NMF_mfeature, NMF_dfeature = get_low_feature(D, 0.01, pow(10, -4), associations)

    for train_index, val_index in kf.split(samples):
        fold += 1
        train_samples = samples[train_index, :]
        print(train_samples.shape)
        val_samples = samples[val_index, :]
        new_association = association.copy()
        for i in val_samples:
            new_association[i[0], i[1]] = 0

        # m_network = sim_thresholding(m_fusion_sim, s[0])
        # m_adj, meta_features = generate_adj_and_feature(m_network, new_association)
        # m_features = get_gae_feature(m_adj, meta_features, s[1], 1)
        # d_features = five_AE(d_fusion_sim)
        # s_features = np.load(r'G:/Python/ED_SDA/NMF/gmodel_feat_sno_fusion.npy')
        s_features = np.load(r'G:/Python/ED_SDA/NMF/gmodel_feat_sno_fusion-2-WKNKN.npy')
        # print(s_features)
        # d_features = np.load(r'G:/Python/ED_SDA/NMF/gmodel_feat_dis_fusion.npy')
        d_features = np.load(r'G:/Python/ED_SDA/NMF/gmodel_feat_dis_fusion-2-WKNKN.npy')

        trans_s = pd.read_csv('./trans_s.csv',sep=' ',header=None).to_numpy()
        # print(trans_s.shape)
        trans_d = pd.read_csv('./trans_d.csv',sep=' ',header=None).to_numpy()
        # print(trans_d.shape)
        train_feature, train_label = generate_f1(D, train_samples, s_features, d_features, NMF_mfeature, NMF_dfeature,trans_d,trans_s)
        val_feature, val_label = generate_f1(D, val_samples, s_features, d_features, NMF_mfeature, NMF_dfeature,trans_d,trans_s)
        print(train_feature.shape)
        print('1')
        # 转换为张量
        train_feature = torch.tensor(train_feature, dtype=torch.float32).to(device)
        train_label = torch.tensor(train_label, dtype=torch.float32).to(device)
        val_feature = torch.tensor(val_feature, dtype=torch.float32).to(device)
        val_label = torch.tensor(val_label, dtype=torch.float32).to(device)

        # 模型、损失函数、优化器
        input_dim = train_feature.shape[1]
        dropout=0
        model = MLP(in_dim=input_dim, hidden_dim=128, hidden_dim1=64, out_dim=1, dropout=dropout)
        # model = KAN(layers_hidden=[218,64,1])
        # model = KAN(layers_hidden=[218,8,1])
        # model = MLPClassifier(input_dim).to(device)
        criterion = nn.BCELoss()
        # optimizer = optim.RMSprop(model.parameters(), lr=0.001)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        current_fold_losses = []
        current_val_losses = []
        # 训练模型
        model.train()
        for epoch in range(classifier_epochs):
            optimizer.zero_grad()
            outputs = model(train_feature).squeeze()
            loss = criterion(outputs, train_label)
            loss.backward()
            optimizer.step()
            # 预测
            with torch.no_grad():
                current_fold_losses.append(loss.item())
                val_outputs = model(val_feature).squeeze()
                val_loss = criterion(val_outputs, val_label)
                current_val_losses.append(val_loss.item())
                y_score = model(val_feature).cpu().numpy()
        # # 保存预测值和对应的非编码RNA和疾病对
        # save_predictions_and_pairs(val_outputs, val_samples, fold)

        fold_losses.append(current_fold_losses)
        val_fold_losses.append(current_val_losses)
        # print(y_score.shape)
        # 计算指标
        fpr, tpr, thresholds = roc_curve(val_label.cpu().numpy(), y_score)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        # 计算 AUPR
        precision, recall, _ = precision_recall_curve(val_label.cpu().numpy(), y_score)
        precisions.append(np.interp(mean_recall, recall[::-1], precision[::-1]))
        precisions[-1][0] = 1.0
        aupr_score = average_precision_score(val_label.cpu().numpy(), y_score)
        auprs.append(aupr_score)
        result += get_metrics(val_label.cpu().numpy(), y_score)
        print('[aupr, auc, f1_score, accuracy, recall, specificity, precision]', get_metrics(val_label.cpu().numpy(), y_score))
