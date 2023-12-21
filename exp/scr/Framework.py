from models import FBMF, LTMF
import numpy as np
import torch
from tqdm import tqdm
from utils import *
from sklearn.metrics import precision_score, precision_recall_curve, auc
import torch


class PCPL():
    # parallel
    def __init__(self, train_mat:np.ndarray, drug_embeddings:torch.Tensor, adr_embeddings:torch.Tensor, valid_set:np.ndarray, thres=0.95, u_thres=0.05) -> None:
        self.train_mat = train_mat
        self.valid_set = valid_set
        self.drug_embeddings = drug_embeddings
        self.adr_embeddings = adr_embeddings
        self.n_models = len(drug_embeddings)
        self.thres = [thres] * self.n_models
        self.u_thres = u_thres
        self.es = EarlyStopping()
        self.weights = [np.ones_like(self.train_mat)] * self.n_models
        self.nums = []
        self.hr = []

    def fit(self, tune=False, test_set=None):
        if tune == True:
            self.models = [FBMF(self.drug_embeddings[i], self.adr_embeddings[i], lr=0.2) for i in range(self.n_models)]
        train_mats = [self.train_mat.copy()] * self.n_models
        for i in tqdm(range(100)):
            scores_list = []
            if tune == False:
                self.models = [FBMF(self.drug_embeddings[i], self.adr_embeddings[i], lr=0.2) for i in range(self.n_models)]

            for j in range(len(self.models)):
                train_mat = train_mats[j]
                model = self.models[j]
                model.fit(torch.from_numpy(train_mat), epochs=50, weight=torch.from_numpy(self.weights[j]), lamb=1e-5)
                scores = model.inference(logit=True)      # logit output
                scores_list.append(scores.numpy())
            train_mats, ps_mats = self.mark_label(scores_list)
            if test_set is not None:
                # 记录伪标签数量和质量的变化过程
                extra_set = np.concatenate([self.valid_set, test_set], axis=0)
                ps_mat = ps_mats[0]
                self.nums.append(np.sum(ps_mat))
                extra_mat = create_matrix(extra_set)
                num_hit = np.sum(ps_mat * extra_mat)
                hit_rate = num_hit / np.sum(ps_mat)
                self.hr.append(hit_rate)
            self.weights = self.dynamic_weight(scores_list, train_mats, ps_mats, t=i, alpha=args.alpha, gamma=args.gamma)
            scores = self.ensemble(scores_list, logit=False)
            topk_auc = self.aupr_measure(scores, k=1.0)

            # print(topk_auc)
            best_scores, _, _ = self.es(topk_auc, scores, thres=self.thres, train_mats=[])
            if self.es.early_stop == True:
                return best_scores

        return best_scores
    
    def ensemble(self, scores_list, logit=False): 
        if logit:
            return sigmoid(np.sum(np.stack(scores_list), axis=0) / len(scores_list))
        else:
            return np.sum(np.stack(scores_list), axis=0) / len(scores_list)
    

    def mark_label(self, scores_list) -> list:

        train_mats = []
        ps_mats = []
        for i in range(len(scores_list)):
            scores = scores_list[i]
            t = self.thres[i]
            mask1 = (scores > t).astype(int)
            a = np.sum(mask1)
            mask2 = 1 - self.train_mat
            b = np.sum(mask2)
            pseudo_labels = mask1 * mask2
            d = np.sum(pseudo_labels)
            train_mat = pseudo_labels + self.train_mat
            train_mats.append(train_mat)
            ps_mats.append(pseudo_labels)

        return train_mats, ps_mats
    

    def neg_cross_entropy(self, train_mat:np.ndarray, scores:np.ndarray):
        idx0, idx1 = np.where(train_mat == 0)
        neg_scores = torch.from_numpy(scores[idx0, idx1]).float()
        neg_targets = torch.from_numpy(train_mat[idx0, idx1]).float()
        ce = torch.nn.functional.binary_cross_entropy(neg_scores, neg_targets)
        return ce.item()

           
    def dynamic_weight(self, scores_list:np.ndarray, trian_mats:np.ndarray, ps_mats:np.ndarray, t, alpha=0.8, gamma=2):
        # alpha 越大，抑制伪标签的作用越大
        weights = []
        for i in range(len(scores_list)):
            scores = scores_list[i]
            pseudo_labels = ps_mats[i]
            trian_mat = trian_mats[i]
            el2n_scores0 = np.abs(scores - self.train_mat)      # 与原数据集的EL2N scores
            el2n_scores1 = np.abs(scores - trian_mat)           # 与新数据集的EL2N scores
            weight =  -1 * (np.log(el2n_scores0 + 1e-9) * (alpha)**gamma + np.log(el2n_scores1 + 1e-9) * (1 - alpha)**gamma)    # weight 
            weight = weight * pseudo_labels
            weight[weight == 0] = (1 / (t+1))
            weights.append(weight)
        return weights

    def metric(self, scores, k):

        valid_mat = create_matrix(self.valid_set)
        scores = (scores * valid_mat).flatten()
        valid = valid_mat.flatten()
        idx = np.argsort(scores)[::-1]
        if isinstance(k, float):
            k = int(len(idx) * k)
        topk_idx = idx[0:k]
        topk_scores = scores[topk_idx]
        topk_labels = valid[topk_idx]
        topk_precision = precision_score(topk_labels, topk_scores)

        return topk_precision
    
    def aupr_measure(self, scores, k):

        # valid_mat = create_matrix(self.valid_set)
        # scores = (scores * valid_mat).flatten()
        # valid = valid_mat.flatten()
        drug_idx = self.valid_set[:,0]
        adr_idx = self.valid_set[:,1]
        labels = self.valid_set[:,2]
        scores = scores[drug_idx, adr_idx]
        idx = np.argsort(scores)[::-1]
        if isinstance(k, float):
            k = int(len(idx) * k)
        topk_idx = idx[0:k]
        topk_scores = scores[topk_idx]
        topk_labels = labels[topk_idx]
        precision, recall, thresholds = precision_recall_curve(topk_labels, topk_scores)
        topk_aupr = auc(recall,precision)
        
        return topk_aupr
    