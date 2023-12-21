import numpy as np
from rdkit import Chem
from sklearn.decomposition import KernelPCA
from rdkit.Chem import AllChem
from param_parser import parameter_parser
from sklearn.metrics import f1_score,precision_recall_curve,auc,precision_score,recall_score,matthews_corrcoef,roc_curve
import pandas as pd

args = parameter_parser()

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

def create_matrix(pairs):

    x, y, lables = pairs[:,0], pairs[:,1], pairs[:,2]
    matirx = np.zeros((args.n_drug, args.n_adr), dtype=np.int32)
    matirx[x, y ] = lables

    return matirx


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


def eval(test_set, pred_mat, thres, t):

    drug_indices = test_set[:,0]
    adr_indices = test_set[:,1]
    labels = test_set[:,2]

    scores = pred_mat[drug_indices, adr_indices]
    scores_ = scores.copy()
    scores_ = (scores_ >= thres).astype('int')
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    data = np.stack([precision, recall]).T
    data_df = pd.DataFrame(data)
    data_df.columns = ['precision', 'recall']
    writer = pd.ExcelWriter(f'..\\result\\pr_curve{t}.xlsx')  #关键2，创建名称为hhh的excel表格
    data_df.to_excel(writer,'page_1',float_format='%.5f')  #关键3，float_format 控制精度，将data_df写到hhh表格的第一页中。若多个文件，可以在page_2中写入
    writer.save()
    aupr = auc(recall,precision)
    f1 = f1_score(labels, scores_)
    prec = precision_score(y_true=labels,y_pred=scores_)
    recall = recall_score(y_true=labels,y_pred=scores_)
    mcc = matthews_corrcoef(y_true=labels,y_pred=scores_)
    mr = mrank(labels, scores)

    return aupr, f1, prec, recall, mcc, mr, scores, scores_


def mrank(y, y_pre):
    index = np.argsort(-y_pre)
    r_label = y[index]
    r_index = np.array(np.where(r_label == 1)) + 1
    reci_sum = np.sum(1 / r_index)
    # reci_rank = np.mean(1 / r_index)
    # mr = np.mean(r_index)
    return reci_sum