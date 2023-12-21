import numpy as np
import torch
from utils import *
import warnings
from param_parser import parameter_parser
from models import LogitMF
from metric import TopK_Metric


args = parameter_parser()
warnings.filterwarnings("ignore")

if __name__ == "__main__":

    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)    
    aupr_, f1_, precison_, recall_, mcc_, mr_ = [],[],[],[],[],[]
    topk_precision_, topk_recall_ = [],[]
    f_rs = open(args.save_path + '\\' + 'log.txt', 'a')

    for fold in range(5):
        print(f'loading data fold{fold}')
        fpt_path = args.root + '\\' + 'structure.txt'
        data_path = args.root + '\\' + f'{fold}'
        train_set_path = data_path + '\\' + 'train_set.txt'
        vaild_set_path = data_path + '\\' + 'valid_set.txt'
        test_set_path = data_path + '\\' + 'test_set.txt'
        kg_path = data_path + '\\' + 'kg.txt'
        save_root = args.save_path + '\\' + f'{fold}'
        train_set = np.loadtxt(train_set_path, dtype=np.int32)
        valid_set = np.loadtxt(vaild_set_path, dtype=np.int32)
        test_set = np.loadtxt(test_set_path, dtype=np.int32)
        kg = np.loadtxt(kg_path, dtype=np.int32)
        kg[:,2] += (args.n_entity + args.n_adr)
        train_set[:,1] -= 2926
        valid_set[:,1] -= 2926
        test_set[:,1] -= 2926
        drug_se_mat = np.loadtxt(args.root + '\\' + 'drug_se_mat.txt')
        train_mat = create_matrix(train_set)

        fpts = load_fpt_vec(fpt_path)
        model = LogitMF(fpts=torch.from_numpy(fpts))
        model.fit(torch.from_numpy(train_mat), lr=0.01, epochs=500)
        scores = model.inference().numpy()

        thres = find_thres(valid_set, scores)
        topk_metric = TopK_Metric(test_set, scores, thres, k=15, n_drug=args.n_drug, n_adr = args.n_adr)
        aupr, f1, prec, recall, mcc, mr, scores, scores_ = eval(test_set, scores, thres=thres, t=fold)
        topk_precision = topk_metric.topk_macro_precison()
        topk_recall = topk_metric.topk_macro_recall()

        topk_precision_.append(topk_precision)
        topk_recall_.append(topk_recall)
        aupr_.append(aupr)
        f1_.append(f1)
        precison_.append(prec)
        recall_.append(recall)
        mcc_.append(mcc)
        mr_.append(mr)
        print(f'aupr: {aupr}   f1: {f1}   topk_precision:{topk_precision}   topk_recall:{topk_recall}')

    aupr = np.around(np.mean(aupr_), 4)
    f1 = np.around(np.mean(f1_), 4)
    precison = np.around(np.mean(precison_), 4)
    recall = np.around(np.mean(recall_), 4)
    mcc = np.around(np.mean(mcc_), 4)
    mr = np.around(np.mean(mr_), 4)
    topk_precision = np.around(np.mean(topk_precision_),4)
    topk_recall = np.around(np.mean(topk_recall_),4)


    print(f'aupr: {aupr}   f1: {f1}   precision: {precison}   recall: {recall}   mcc: {mcc}  mr: {mr}   topk_recall:{topk_recall}   topk_precision:{topk_precision}')