from param_parser import parameter_parser
import numpy as np
from utils import *
from models import LinearModel
from scipy import linalg
from metric import TopK_Metric
import warnings
warnings.filterwarnings("ignore")


def cal_kernels(train_mat, mode):

    if mode == 'drug':
        fpt_path = args.root + '\\' + 'structure.txt'
        fpts = create_fpts_matrix(fpt_path)
        gip_d_c = GIP_kernel(fpts)
        mi_d_c = MI_kernel(fpts)
        cos_d_c = COS_kernel(fpts)
        corr_d_c = CORR_kernel(fpts)

        # Kernels for drugs
        gip_d_l = GIP_kernel(train_mat)
        mi_d_l = MI_kernel(train_mat)
        cos_d_l = COS_kernel(train_mat)
        corr_d_l = CORR_kernel(train_mat)
        ideal_d = ideal_Kernel(train_mat)

        return [gip_d_c, mi_d_c, cos_d_c, corr_d_c, gip_d_l, mi_d_l, cos_d_l, corr_d_l], ideal_d
    else:
        # Kernels for side-effects
        gip_s_l = GIP_kernel(train_mat)
        mi_s_l = MI_kernel(train_mat)
        cos_s_l = COS_kernel(train_mat)
        corr_s_l = CORR_kernel(train_mat)
        ideal_s = ideal_Kernel(train_mat)

        return [gip_s_l, mi_s_l, cos_s_l, corr_s_l], ideal_s

if __name__ == "__main__":

    seed = 123
    np.random.seed(seed)
    args = parameter_parser()

    aupr_, f1_, precison_, recall_, mcc_, mr_ = [],[],[],[],[],[]
    topk_precision_, topk_recall_ = [],[]

    for fold in range(5):
        # print(f'loading data fold{fold}')
        # fpt_path = args.root + '\\' + 'structure.txt'
        data_path = args.root + '\\' + f'{fold}'
        # train_set_path = data_path + '\\' + 'train_set.txt'
        vaild_set_path = data_path + '\\' + 'valid_set.txt'
        test_set_path = data_path + '\\' + 'test_set.txt'
        # train_set = np.loadtxt(train_set_path, dtype=np.int32)
        valid_set = np.loadtxt(vaild_set_path, dtype=np.int32)
        test_set = np.loadtxt(test_set_path, dtype=np.int32)
        # train_set[:,1] -= 2926
        valid_set[:,1] -= 2926
        test_set[:,1] -= 2926
        drug_se_mat = np.loadtxt(args.root + '\\' + 'drug_se_mat.txt')
        # train_mat = create_matrix(train_set)

        # print('caculating kernels...')
        # kernels_d, ideal_kernel_d = cal_kernels(train_mat, mode='drug')
        # kernels_s, ideal_kernel_s = cal_kernels(train_mat.T, mode='se')

        # # 计算线性权重
        # L_d = LinearModel(len(kernels_d))
        # L_s = LinearModel(len(kernels_s))
        # print('training Linear Model for drug kernels...')
        # loss_d = L_d.fit(kernels_d, ideal_kernel_d)
        # print('training Linear Model for side-effect kernels...')
        # loss_s = L_s.fit(kernels_s, ideal_kernel_s)
        # kernel_d = L_d.combination(kernels_d)
        # kernel_s = L_s.combination(kernels_s)

        # kernel_path = f'..\\result\\{fold}'
        # mkdir(kernel_path)
        # np.savetxt(kernel_path + '\\' + 'kernel_d.txt', kernel_d)
        # np.savetxt(kernel_path + '\\' + 'kernel_s.txt', kernel_s)
        # # kernel_d = np.loadtxt(kernel_path + '\\' + 'kernel_d.txt')
        # # kernel_s = np.loadtxt(kernel_path + '\\' + 'kernel_s.txt')

        # print('running WKNKN...')
        # F_train = WKNKN(train_mat, kernel_d, kernel_s, k=17)
        # np.savetxt(kernel_path + '\\' + 'F_train.txt', F_train)
        # # F_train = np.loadtxt(kernel_path + '\\' + 'F_train.txt')


        # print('running Graph-based Semi-supervised Learning...')
        # Nd = GSL(kernel_d)
        # Ns = GSL(kernel_s)


        # print('running Local and global consistecy algorithm...')
        # kernel_d = kernel_d * Nd
        # kernel_s = kernel_s * Ns

        # Dd = np.diag(np.sum(kernel_d, axis=1))
        # Ds = np.diag(np.sum(kernel_s, axis=1))
        # delta_d = Dd - kernel_d
        # delta_s = Ds - kernel_s

        # Dd = np.linalg.inv(Dd)
        # Ds = np.linalg.inv(Ds)
        # Dd_norm = np.sqrt(Dd)
        # Ds_norm = np.sqrt(Ds)
        # Dd_norm = np.nan_to_num(Dd_norm)
        # Ds_norm = np.nan_to_num(Ds_norm)


        # Ld = np.matmul(np.matmul(Dd_norm, delta_d), Dd_norm)
        # Ls = np.matmul(np.matmul(Ds_norm, delta_s), Ds_norm)


        # print('sovling Sylvester equation...')
        # A = np.eye(Ld.shape[0], Ld.shape[1]) + Ld * args.u
        # B = Ls * args.v
        # C = F_train
        # X = linalg.solve_sylvester(A, B, C)

        # print('saving new drug side-effect assosiation...')
        save_path = args.save_path + '\\' + f'result{fold}.txt'
        # np.savetxt(save_path, X)

        print('testing...')
        Y = np.loadtxt(save_path)
        eval_aupr, eval_f1, eval_precision, eval_recall, eval_mcc, eval_mr, best_thres = eval(Y, valid_set, thres=None)
        aupr, f1, precision, recall, mcc, mr, scores, scores_ = eval(Y, test_set, thres=best_thres, t=fold)
        topk_metric = TopK_Metric(test_set, Y, best_thres, k=15, n_drug=args.n_drug, n_adr = args.n_adr)
        topk_precision = topk_metric.topk_macro_precison()
        topk_recall = topk_metric.topk_macro_recall()
        # show_scatter_dicrim(scores, scores_, test_set, drug_se_mat, show=True)
        # positive_count(scores_, test_set, drug_se_mat, show=True)

        print(f'valid{fold}:aupr:{eval_aupr}  f1:{eval_f1}  precision:{eval_precision}  recall:{eval_recall}  mcc:{eval_mcc}  mr:{eval_mr}  threshold:{best_thres}')
        print(f'test{fold}:aupr:{aupr}  f1:{f1}  precision:{precision}  recall:{recall}  mcc:{mcc}  mr:{mr}  threshold:{best_thres}')
        print(f'test{fold}:topk_precision:{topk_precision}  topk_recall:{topk_recall}')
        aupr_.append(aupr)
        f1_.append(f1)
        precison_.append(precision)
        recall_.append(recall) 
        mcc_.append(mcc)
        mr_.append(mr)
        topk_precision_.append(topk_precision)
        topk_recall_.append(topk_recall)

        log_path = args.save_path + '\\' + 'log.txt'
        f = open(log_path, 'a')
        line = f'Fold{fold}   aupr:{aupr}  f1:{f1}  precision:{precision}  recall:{recall}  mcc:{mcc}  mr:{mr}  topk_recall:{topk_recall}  topk_precision:{topk_precision}  thres:{best_thres} \n'
        f.write(line)
        f.close()

    aupr = np.mean(aupr_)
    f1 = np.mean(f1_)
    precision = np.mean(precison_)
    recall = np.mean(recall_)
    mcc = np.mean(mcc_)
    mr = np.mean(mr_)
    topk_precision = np.mean(topk_precision_)
    topk_recall = np.mean(topk_recall_)
    print('===========================================AVERAGE PERFORMANCE============================================')
    print(f'aupr:{aupr}  f1:{f1}  precision:{precision}  recall:{recall}  mcc:{mcc}  mr:{mr}  topk_recall:{topk_recall}  topk_precision:{topk_precision}')
    log_path = args.save_path + '\\' + 'log.txt'
    f = open(log_path, 'a')
    line = f'AVERAGE   aupr:{aupr}  f1:{f1}  precision:{precision}  recall:{recall}  mcc:{mcc}  mr:{mr}  topk_recall:{topk_recall}  topk_precision:{topk_precision}  thres:{best_thres} \n'
    f.write(line)
    f.write('==============================================================================================================================\n')
    f.close()