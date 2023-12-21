import numpy as np
from param_parser import parameter_parser
from tqdm import tqdm
from sklearn.metrics import f1_score,precision_recall_curve,auc,precision_score,recall_score,matthews_corrcoef, roc_curve
import os
import matplotlib.pyplot as plt
import pandas as pd

args = parameter_parser()

def create_matrix(pairs):

    x, y, lables = pairs[:,0], pairs[:,1], pairs[:,2]
    matirx = np.zeros((args.n_drug, args.n_adr), dtype=np.int32)
    matirx[x, y ] = lables

    return matirx

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

def GIP_kernel(mat, gamma=1):

    n = mat.shape[0]
    gip_sim = np.zeros((n,n))
    for i in tqdm(range(n), desc='gip'):
        Fd = mat[i]
        # Fd = np.stack([Fd] * n)
        tmp = np.sqrt(np.sum(np.square(Fd - mat), axis=1)) * -1 * gamma
        gip = np.exp(tmp)
        gip_sim[i] = gip
    gip_sim = np.nan_to_num(gip_sim)

    return gip_sim

def CORR_kernel(mat):

    n = mat.shape[0]
    corr_sim = np.zeros((n,n))
    for i in tqdm(range(n), desc='corr'):
        Fd = mat[i]
        corr = np.corrcoef(Fd, mat)
        tmp = corr[0,1:]
        corr_sim[i] = tmp

    corr_sim = np.nan_to_num(corr_sim)
    return corr_sim

def COS_kernel(mat):

    n = mat.shape[0]
    cos_sim = np.zeros((n,n))
    for i in tqdm(range(n), desc='cosine'):
        Fd = mat[i]
        cos = np.sum(Fd * mat, axis=1) / (np.linalg.norm(Fd) * np.linalg.norm(mat))
        cos_sim[i] = cos

    cos_sim = np.nan_to_num(cos_sim)
    return cos_sim

def MI_kernel(mat):

    n = mat.shape[0]
    MI_sim = np.zeros((n,n))
    for i in tqdm(range(n), desc='MI'):
        Fd = mat[i]
        mi = binary_mutula_information(Fd, mat)
        MI_sim[i] = mi

    MI_sim = np.nan_to_num(MI_sim)
    return MI_sim

def normalized(mat):

    diag = np.diag(mat) # 对角元素
    z = np.sqrt(np.outer(diag, diag))
    mat_normalized = mat / z
    mat_normalized = np.nan_to_num(mat_normalized)
    return mat_normalized

def ideal_Kernel(mat):

    ideal = np.matmul(mat, mat.T)
    ideal = normalized(ideal)

    return ideal


def binary_mutula_information(x, y):

    '''
    x.shape: (n,)
    y.shape: (m, n)
    x -> [0,1,0]
    y -> [[1,0,0],[1,0,0]]
    return [mi, mi]
    '''
    px1 = np.sum(x) / x.shape[0]
    px0 = 1 - px1
    py1 = np.sum(y, axis=1) / y.shape[1]
    py0 = 1 - py1
    tmp_mat1 = x - y    # x=1而 y=0时，元素为1；y=1而x=0时，元素为-1，其余为0
    pxy10 = np.sum(tmp_mat1 == 1, axis=1) / y.shape[1]
    pxy01 = np.sum(tmp_mat1 == -1, axis=1) / y.shape[1]
    tmp_mat2 = x * y    # x=1而y=1时，元素为1， 其余为0
    pxy11 = np.sum(tmp_mat2, axis=1) / y.shape[1]
    pxy00 = 1 - (pxy01 + pxy11 + pxy10) # 剩余的为全时0的概率

    mi = 0.0
    # 累加00，01，10，00四种情况的MI值
    mi += pxy00 * np.log(pxy00 / (px0 * py0))
    mi = np.nan_to_num(mi)
    mi += pxy11 * np.log(pxy11 / (px1 * py1))
    mi = np.nan_to_num(mi)
    mi += pxy01 * np.log(pxy01 / (px0 * py1))
    mi = np.nan_to_num(mi)
    mi += pxy10 * np.log(pxy10 / (px1 * py0))
    mi = np.nan_to_num(mi)

    return mi

def WKNKN(drug_se_mat, drug_sim_mat, se_sim_mat, k=17):

    Yd, Ys = np.zeros_like(drug_se_mat, dtype=np.float32), np.zeros_like(drug_se_mat, dtype=np.float32)
    n_drug, n_se = drug_se_mat.shape
    ang = np.linspace(1.0, 0.0, num=k)
    for d in tqdm(range(n_drug)):
        sim = drug_sim_mat[d]   #药物d与其他药物的相似性
        sim[d] = 0  # 药物d与自身的相似性为0
        indices = np.argsort(-sim)    # 降序排序
        sim = sim[indices[0:k]]      # 前k个药物的相似性
        Zd = np.sum(sim)    # Norm term
        w = (sim * ang).reshape(-1,1)
        sim_drug_se_info = drug_se_mat[indices[0:k]]      # 获取前k个最相似药物的药物不良反应信息
        tmp = np.sum(sim_drug_se_info * w, axis=0) / Zd
        Yd[d] = tmp
        a = max(Yd[d])
    Yd = np.nan_to_num(Yd)

    drug_se_mat_ = drug_se_mat.T
    for s in tqdm(range(n_se)):
        sim = se_sim_mat[s]
        sim[s] = 0
        indices = np.argsort(-sim)    # 降序排序
        sim = sim[indices[0:k]]      # 前k个不良反应的相似性
        Zs = np.sum(sim)    # Norm term
        w = (sim * ang).reshape(-1,1)
        sim_se_drug_info = drug_se_mat_[indices[0:k]]      # 获取前k个最相似药物的药物不良反应信息
        tmp = np.sum(sim_se_drug_info * w, axis=0) / Zs
        Ys[:,s] = tmp
    Ys = np.nan_to_num(Ys)

    Yds = (Yd + Ys) / 2
    mask1 = (drug_se_mat > Yds).astype(int)
    mask2 = 1 - mask1
    rs = drug_se_mat * mask1 + Yds * mask2

    return rs

def GSL(kernel, k=17):

    n = kernel.shape[0]
    mat = np.zeros_like(kernel)
    for i in tqdm(range(n)):
        sim = kernel[i]
        sim[i] = 0
        indices = np.argsort(-sim)    # 降序排序
        indices[0:k]
        mat[i,indices] += 0.5
        mat[indices, i] += 0.5
    
    return mat


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

def eval(Y, test_set, thres=None, t=0):

    x = test_set[:,0]
    y = test_set[:,1]
    labels = test_set[:,2]
    scores = Y[x, y]
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    data = np.stack([precision, recall]).T
    data_df = pd.DataFrame(data)
    data_df.columns = ['precision', 'recall']
    writer = pd.ExcelWriter(f'..\\result\\pr_curve{t}.xlsx')  #关键2，创建名称为hhh的excel表格
    data_df.to_excel(writer,'page_1',float_format='%.5f')  #关键3，float_format 控制精度，将data_df写到hhh表格的第一页中。若多个文件，可以在page_2中写入
    writer.save()
    aupr = auc(recall,precision)
    best_f1 = 0
    best_thres = 0
    if thres is None:
        thresholds = list(thresholds)
        thresholds = thresholds[0:len(thresholds):int(len(thresholds)/1000)]
        for i in tqdm(range(len(thresholds))):
            scores_ = scores.copy()
            scores_ = (scores_ >= thresholds[i]).astype('int')
            f1 = f1_score(y_true=labels,y_pred=scores_)
            if best_f1 <= f1:
                best_f1 = f1
                best_thres = thresholds[i]

        precision = precision_score(labels, scores_)
        recall = recall_score(labels, scores_)
        mcc = matthews_corrcoef(labels, scores_)
        f1 = f1_score(y_true=labels,y_pred=scores_)
        mr = mrank(labels, scores)

        return aupr, f1, precision, recall, mcc, mr, best_thres
    else:
        scores_ = scores.copy()
        scores_ = (scores_ >= thres).astype('int')
        precision = precision_score(labels, scores_)
        recall = recall_score(labels, scores_)
        mcc = matthews_corrcoef(labels, scores_)
        f1 = f1_score(y_true=labels,y_pred=scores_)
        mr = mrank(labels, scores)

        return aupr, f1, precision, recall, mcc, mr, scores, scores_
    

def mkdir(path):

    folder = os.path.exists(path)

    if not folder: #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path) #makedirs 创建文件时如果路径不存在会创建这个路径


def contain_nan(x):
    contain_nan = (True in np.isnan(x))
    return contain_nan

def show_scatter_dicrim(scores, scores_, input, train_mat, b=None, show=False):

    labels = input[:,2]
    freq = np.sum(train_mat, axis=0)
    sorted_indices = np.argsort(freq)[::-1]
    freq = np.sort(freq)[::-1] 
    # sorted_indices = sorted_indices.numpy()
    x_c, y_c = [],[]
    x_f, y_f = [],[]
    # sorted_indices = sorted_indices.numpy()
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


def positive_count(scores_, test_data, train_mat, b=None, show=False):

    rs = []
    # train_mat = torch.from_numpy(drug_se_mat)
    freq = np.sum(train_mat, axis=0)      # number of each adr
    # freq, sorted_indices = np.sort(freq, descending=True)
    sorted_indices = np.argsort(freq)[::-1]
    freq = np.sort(freq)[::-1] 
    # sorted_indices = sorted_indices.numpy()
    adr_indices = test_data[:,1]
    for i in sorted_indices:
        count = (adr_indices == i).astype(int)
        pos_count = count * scores_
        pos_count = np.sum(pos_count)
        rs.append(pos_count)

    x = [i for i in range(len(rs))]
    plt.figure(figsize=(16,8))
    plt.title('number of positive instances after prediction')
    plt.bar(x,rs)
    plt.ylabel('Frequency')
    plt.xlabel('Sorted Class index')
    if b is not None:
        plt.savefig(f'{b}(1)'+'.png')
    if show == True:
        plt.show()
    plt.close()
    return rs


# def mrank(y, y_pre):
#     index = np.argsort(-y_pre)
#     r_label = y[index]
#     r_index = np.array(np.where(r_label == 1)) + 1
#     reci_sum = np.sum(1 / r_index)
#     mr = np.mean(r_index)
#     return mr
def mrank(y, y_pre):
    index = np.argsort(-y_pre)
    r_label = y[index]
    r_index = np.array(np.where(r_label == 1)) + 1
    reci_sum = np.sum(1 / r_index)
    # reci_rank = np.mean(1 / r_index)
    # mr = np.mean(r_index)
    return reci_sum