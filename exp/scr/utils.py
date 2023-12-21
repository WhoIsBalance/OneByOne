import numpy as np
import json
import networkx as nx
from param_parser import parameter_parser
import os
from tqdm import tqdm
from sklearn.metrics import f1_score,precision_recall_curve,auc,precision_score,recall_score,matthews_corrcoef,roc_curve
from sklearn import preprocessing
import torch
import matplotlib.pyplot as plt
from rdkit import Chem
from sklearn.decomposition import KernelPCA
from rdkit.Chem import AllChem
import pandas as pd

args = parameter_parser()

def read_config(model_name):
    """"读取配置"""
    path = '..\\configs\\' + f'{model_name}' + "config.json"
    with open(path) as json_file:
        config = json.load(json_file)
    return config

def create_fpts_matrix(path):

    f = open(path, 'r')
    lines = f.readlines()
    f.close()

    mat = []
    for line in lines:
        pieces = line.strip().split('\t')
        fpt = pieces[2]
        mat.append(list(fpt))
    
    mat = np.array(mat, dtype=np.int32)
    return mat

def create_graph(kg):

    G = nx.Graph()
    kg = kg.tolist()
    for node in range(args.n_entity + args.n_adr + args.n_relation):
        G.add_node(node)
    for line in kg:
        # G.add_edges_from([(line[0],line[1]),(line[1],line[2])])
        G.add_edges_from([(line[0],line[2]),(line[2],line[1])])

    return G

def create_matrix(pairs):

    x, y, lables = pairs[:,0], pairs[:,1], pairs[:,2]
    matirx = np.zeros((args.n_drug, args.n_adr), dtype=np.int32)
    matirx[x, y ] = lables

    return matirx



def mkdir(path):

    if not os.path.exists(path):
        os.mkdir(path)

def load_vec(path):

    vec = np.loadtxt(path)
    # vec = vec[np.argsort(vec[:,0])][:,1:]
    vec = vec[np.argsort(vec[:,0])]
    return vec[:,1:]


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def find_thres(valid_set, pred_mat, metric='f1'):

    def f1(p, r):
        if p + r != 0:
            return 2 * (p * r) / (p + r)
        else:
            return 0
    def gmean(tpr, fpr):
        return np.sqrt(tpr * (1 - fpr))
    
    def youden(tpr, fpr):
        return tpr - fpr

    drug_indices = valid_set[:,0]
    adr_indices = valid_set[:,1]
    labels = valid_set[:,2]
    scores = pred_mat[drug_indices, adr_indices]

    # f1s=[f1(p, r) for p, r in zip(precisions, recalls)]
    if metric == 'f1':
        precisions, recalls, thresholds = precision_recall_curve(labels, scores)
        f1s=[f1(p, r) for p, r in zip(precisions, recalls)]
        optimal_index = np.argmax(f1s)
        
    elif metric == 'gmean':
        fpr, tpr, thresholds = roc_curve(labels, scores)
        gmeans = gmean(tpr, fpr)
        optimal_index = np.argmax(gmeans)

    elif metric == 'youden':
        fpr, tpr, thresholds = roc_curve(labels, scores)
        youdens = youden(tpr, fpr)
        optimal_index = np.argmax(youdens)
        
    optimal_threshold = thresholds[optimal_index]

    return optimal_threshold

def show_scatter_dicrim(scores, scores_, input, drug_se_mat, b=None, show=False):

    labels = input[:,2]
    # input = torch.from_numpy(input)
    train_mat = torch.from_numpy(drug_se_mat)
    freq = torch.sum(train_mat, dim=0)      # number of each adr
    freq, sorted_indices = torch.sort(freq, descending=True)
    x_c, y_c = [],[]
    x_f, y_f = [],[]
    sorted_indices = sorted_indices.numpy()
    for i in range(len(sorted_indices)):
        indices = np.where(input[:,1] == sorted_indices[i])[0]
        for j in range(len(indices)):
            index = int(indices[j])
            if scores_[index] == labels[index]:
                x_c.append(i)
                y_c.append(float(scores[index]))
            else:
                y_f.append(float(scores[index]))
                x_f.append(i)

    
    plt.figure(figsize=(16,8))
    plt.title('Result of classifier')
    plt.scatter(x_c,y_c,s=0.5,c='r',label='correct')
    plt.scatter(x_f,y_f,s=0.5,c='g',label='false')
    plt.ylabel('Score')
    plt.xlabel('Sorted Class index')
    plt.legend(loc='best')
    if b is not None:
        plt.savefig(b + '.png')
    if show == True:
        plt.show()
    plt.close()


def getTanimotocoefficient(s,t):
    # 计算谷本系数
    # s=np.asarray(s)
    # t=np.asarray(t)
    if (s.shape!=t.shape):
        print("向量长度不一致")
        return np.nan
    return (np.sum(s*t))/(np.sum(s**2)+np.sum(t**2)-np.sum(s*t))

def get_sim_mat(data):
    sim_mat = np.zeros((len(data), len(data)))
    for i in tqdm(range(len(data))):
        fp1 = data[i]
        for j in range(i, len(data)):
            fp2 = data[j]
            sim = getTanimotocoefficient(fp1, fp2)
            sim_mat[i,j] = sim
            sim_mat[j,i] = sim
    sim_mat = np.nan_to_num(sim_mat)

    return sim_mat

def load_sturcture(path):
    f = open(path, 'r')
    lines = f.readlines()
    f.close()

    data = []
    for line in lines:
        contents = line.strip().split('\t')
        data.append(list(contents[2]))
    data = np.array(data, dtype=np.int32)

    return data


def eval(test_set, pred_mat, thres, t):

    drug_indices = test_set[:,0]
    adr_indices = test_set[:,1]
    labels = test_set[:,2]

    scores = pred_mat[drug_indices, adr_indices]
    scores_ = scores.copy()
    scores_ = (scores_ >= thres).astype('int')
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    # data = np.stack([precision, recall]).T
    # data_df = pd.DataFrame(data)
    # data_df.columns = ['precision', 'recall']
    # writer = pd.ExcelWriter(f'..\\result\\pr_curve{t}.xlsx')  #关键2，创建名称为hhh的excel表格
    # data_df.to_excel(writer,'page_1',float_format='%.5f')  #关键3，float_format 控制精度，将data_df写到hhh表格的第一页中。若多个文件，可以在page_2中写入
    # writer.save()
    aupr = auc(recall,precision)
    f1 = f1_score(labels, scores_)
    prec = precision_score(y_true=labels,y_pred=scores_)
    recall = recall_score(y_true=labels,y_pred=scores_)
    mcc = matthews_corrcoef(y_true=labels,y_pred=scores_)
    mr = mrank(labels, scores)

    return aupr, f1, prec, recall, mcc, mr, scores, scores_


def drop(t: np.ndarray, p:float=0.2):

    pos_x, pos_y = np.where(t > 0)
    idx = np.random.choice(len(pos_x), int(t.sum() * p))

    x = pos_x[idx]
    y = pos_y[idx]
    mask = np.ones_like(t, dtype=np.float32)
    mask[x, y] = 0

    return mask * t


def nosiy(train_mat, ratio=0.5):

    mat = np.zeros_like(train_mat, dtype=np.float32)
    l = mat.shape[0]*mat.shape[1]
    idx = np.random.choice(l, size=int(l*ratio))
    x = idx // mat.shape[1]
    y = idx % mat.shape[1]
    mat[x, y] = 1
    return mat


def mrank(y, y_pre):
    index = np.argsort(-y_pre)
    r_label = y[index]
    r_index = np.array(np.where(r_label == 1)) + 1
    reci_sum = np.sum(1 / r_index)
    # reci_rank = np.mean(1 / r_index)
    # mr = np.mean(r_index)
    return reci_sum


def softmax( f ):
    # instead: first shift the values of f so that the highest number is 0:
    f -= np.max(f) # f becomes [-666, -333, 0]
    return np.exp(f) / np.sum(np.exp(f))  # safe to do, gives the correct answer


def ensemble(scores_list):
    result = 0
    for score in scores_list:
        result += score
    return result / len(scores_list)

def neg_weight(train_mat, p, a):

    zy = np.sum(train_mat) / (train_mat.shape[0] * train_mat.shape[1])
    tmp = (np.exp(zy) - 1) ** p
    weight = a * tmp / (tmp * np.sum(train_mat))

    return weight

def density(train_mat:np.ndarray, norm=False):
    a = np.sum(train_mat, axis=0, keepdims=True)
    b = np.sum(train_mat, axis=1, keepdims=True)
    a = np.concatenate([a] * train_mat.shape[0])
    b = np.concatenate([b] * train_mat.shape[1], axis=1)

    if norm == False:
        return a + b
    else:
        result = a + b
        min_max_scaler = preprocessing.MinMaxScaler((0, 1))
        result = min_max_scaler.fit_transform(result)

def BALD(pred_mats:list, pred_mat:np.ndarray) -> np.ndarray:
    T = len(pred_mats)  # number of classifier
    pred_mat = torch.from_numpy(pred_mat)
    item1 = torch.sigmoid(pred_mat) * torch.nn.LogSigmoid()(pred_mat)
    item2 = 0
    for p in pred_mats:
        p = torch.from_numpy(p)
        item2 += (torch.sigmoid(p) * torch.nn.LogSigmoid()(p))
    item2 /= T
    return (item2 - item1).numpy()


class EarlyStopping():

    def __init__(self, patience=3, delta=0, mode='g') -> None:
        
        self.patience = patience
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.best_result = {'pred':[], 'score':[], 'thres':[], 'train':[]}
        self.mode = mode
        self.counter = 0

    def __call__(self, score, pred, thres, train_mats):
        
        if self.mode == 'g':
            if self.best_score is None:
                self.best_score = score
                self.best_result['pred'].append(pred)
                self.best_result['score'].append(score)
                self.best_result['thres'].append(thres)
                self.best_result['train'].append(train_mats)
            elif score < self.best_score + self.delta:
                self.counter += 1
                self.best_result['pred'].append(pred)
                self.best_result['score'].append(score)
                self.best_result['thres'].append(thres)
                self.best_result['train'].append(train_mats)
                # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
                    idx = self.best_result['score'].index(max(self.best_result['score']))
                    return self.best_result['pred'][idx], self.best_result['thres'][idx], self.best_result['train'][idx]
            else:
                self.best_score = score
                self.counter = 0
                self.best_result = {'pred':[], 'score':[], 'thres':[], 'train':[]}
                self.best_result['pred'].append(pred)
                self.best_result['score'].append(score)
                self.best_result['thres'].append(thres)
                self.best_result['train'].append(train_mats)
        else:
            if self.best_score is None:
                self.best_result['pred'].append(pred)
                self.best_result['score'].append(score)
                self.best_result['thres'].append(thres)
                self.best_result['train'].append(train_mats)
                self.best_score = score
            elif score > self.best_score + self.delta:
                self.counter += 1
                self.best_result['pred'].append(pred)
                self.best_result['score'].append(score)
                self.best_result['thres'].append(thres)
                self.best_result['train'].append(train_mats)
                # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
                    idx = self.best_result['score'].index(max(self.best_result['score']))
                    return self.best_result['pred'][idx], self.best_result['thres'][idx], self.best_result['train'][idx]
            else:
                self.best_score = score
                self.counter = 0
                self.best_result = {'pred':[], 'score':[], 'thres':[], 'train':[]}
                self.best_result['pred'].append(pred)
                self.best_result['score'].append(score)
                self.best_result['thres'].append(thres)
                self.best_result['train'].append(train_mats)
    
        return pred, thres, train_mats
    


def getTanimotocoefficient(s,t):
    # 计算谷本系数
    # s=np.asarray(s)
    # t=np.asarray(t)
    if (s.shape!=t.shape):
        print("向量长度不一致")
        return np.nan
    return (np.sum(s*t))/(np.sum(s**2)+np.sum(t**2)-np.sum(s*t))

def get_sim_mat(data) -> np.ndarray:
    sim_mat = np.zeros((len(data), len(data)))
    for i in tqdm(range(len(data))):
        fp1 = data[i]
        for j in range(i, len(data)):
            fp2 = data[j]
            sim = getTanimotocoefficient(fp1, fp2)
            sim_mat[i,j] = sim
            sim_mat[j,i] = sim
    sim_mat = np.nan_to_num(sim_mat)

    return sim_mat

def load_sturcture(path):
    f = open(path, 'r')
    lines = f.readlines()
    f.close()

    data = []
    for line in lines:
        contents = line.strip().split('\t')
        data.append(contents[1])
    # data = np.array(data, dtype=np.int32)

    return data


def load_fpt_vec(path):
    f = open(path, 'r')
    lines = f.readlines()
    f.close()

    data = []
    for line in lines:
        contents = line.strip().split('\t')
        data.append(contents[1])
    mols = [Chem.MolFromSmiles(canonical_smiles) for canonical_smiles in data]
    data_fps = [list(AllChem.GetMorganFingerprintAsBitVect(x,3,2048).ToBitString()) for x in mols]  # 2048维ECFP摩根指纹
    X = np.array(data_fps, dtype=np.int32)
    scikit_kpca = KernelPCA(n_components=100, kernel='rbf', gamma=15)
    desc = scikit_kpca.fit_transform(X)

    return desc


def predict_topk(k, save_path, drug_se_mat:np.ndarray, scores:np.ndarray):

    adr2id_path = r'D:\Document\Paper1\dataset\final_dataset1\id2adr.json'
    drug2id_path = r'D:\Document\Paper1\dataset\final_dataset1\id2drug.json'
    f_adr = open(adr2id_path, 'r')
    f_drug = open(drug2id_path, 'r')
    id2adr = json.load(f_adr)
    id2drug = json.load(f_drug)
    f_adr.close()
    f_drug.close()


    drug_se_mat = torch.from_numpy(drug_se_mat)
    pred_mat = torch.from_numpy(scores)

    neg_mask = torch.abs(drug_se_mat - 1)
    pred_mat = (pred_mat * neg_mask).flatten()
    freq, sorted_indices = torch.sort(pred_mat, descending=True)

    sorted_indices = sorted_indices.numpy()[0:k]
    lables = np.zeros_like(sorted_indices)
    mat_indices = idx2pair(sorted_indices, lables)
    drug_indices = mat_indices[:,0]
    adr_indices = mat_indices[:,1]

    f = open(save_path, 'w')
    for i in range(k):
        drug = id2drug[str(drug_indices[i])]
        adr = id2adr[str(adr_indices[i])]
        line = drug + '\t' + adr + '\n'
        f.write(line)
    f.close()


def idx2pair(indices, labels):

    y_indices = indices % args.n_adr
    x_indices = (indices / args.n_adr).astype(int)
    out = np.stack([x_indices, y_indices, labels], axis=0)

    return out.T.astype(int)