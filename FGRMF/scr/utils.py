import numpy as np
from sklearn.metrics import f1_score,precision_recall_curve,auc,precision_score,recall_score,matthews_corrcoef,roc_curve
from utils import *
from param_parser import parameter_parser
import pandas as pd

args = parameter_parser()
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
    writer = pd.ExcelWriter(f'pr_curve{t}.xlsx')  #关键2，创建名称为hhh的excel表格
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

def create_matrix(pairs):

    x, y, lables = pairs[:,0], pairs[:,1], pairs[:,2]
    matirx = np.zeros((args.n_drug, args.n_adr), dtype=np.int32)
    matirx[x, y ] = lables

    return matirx


class Strcture_loader:
    '''
    indices,fps,dim_fp
    '''
    def __init__(self,args) -> None:
        self.args = args
        self.indices,self.fps,self.dim_fp = self.load_structure()

    def load_structure(self):
        print('loading structure')
        f = open(r'D:\Document\Paper4\dataset\final_dataset1\structure.txt','r',encoding='utf-8')
        lines = f.readlines()
        f.close()

        indices,fps = [],[]
        for line in lines:
            index,smile,fp = line.strip().split('\t')
            if (fp != 'None'):
                indices.append(int(index))
                fps.append([int(c) for c in fp])
            else:
                fps.append([0] * 881)

        return indices,fps,len(fps[0])
    
def load_structure():
    print('loading structure')
    p = r'D:\Document\Paper4\dataset\final_dataset1\structure.txt'
    f = open(p,'r',encoding='utf-8')
    lines = f.readlines()
    f.close()

    indices,fps = [],[]
    for line in lines:
        index,smile,fp = line.strip().split('\t')
        if (fp != 'None'):
            indices.append(int(index))
            fps.append([int(c) for c in fp])
        else:
            fps.append([0] * 881)

    return indices,fps,len(fps[0])

def load_protein():

    print('loading protein')
    fps = np.loadtxt(r'D:\Document\Paper4\dataset\final_dataset1\kg.txt', dtype=np.int32)
    mat = np.zeros((args.n_drug, args.n_protein))
    for i in range(fps.shape[0]):
        drug_idx, protein_idx = fps[i][0], fps[i][1]
        mat[drug_idx, protein_idx] = 1
    return mat, mat.shape[1]

def load_indication():

    print('loading indication')
    fps = np.loadtxt(args.indication_mat)
    return fps, fps.shape[1]


def mini_batches(X,Y,mini_batch_size=64,seed=0):
    np.random.seed(seed)
    m = X.shape[0] #m是样本数
    
    mini_batches = [] #用来存放一个一个的mini_batch
    permutation = list(np.random.permutation(m)) #打乱标签
    shuffle_X = X[permutation,:] #将打乱后的数据重新排列
    shuffle_Y = Y[permutation]
    
    
    num_complete_minibatches = int(m //mini_batch_size) #样本总数除以每个batch的样本数量
    for i in range(num_complete_minibatches):
        mini_batch_X = shuffle_X[i*mini_batch_size:(i+1)*mini_batch_size,:]
        mini_batch_Y = shuffle_Y[i*mini_batch_size:(i+1)*mini_batch_size]
        mini_batch = (mini_batch_X,mini_batch_Y)
        mini_batches.append(mini_batch)
        
    if m%mini_batch_size != 0:
    	#如果样本数不能被整除，取余下的部分
        mini_batch_X = shuffle_X[num_complete_minibatches*mini_batch_size:,:]
        mini_batch_Y = shuffle_Y[num_complete_minibatches*mini_batch_size:]
        mini_batch = (mini_batch_X,mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches