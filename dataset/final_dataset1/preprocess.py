import numpy as np
import json
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold

if __name__ == "__main__":
    # f = open('new_kg.txt', 'r')
    # lines = f.readlines()
    # f.close()
    # drug_protein_mat = np.zeros((1177, 1749))
    # count = 1
    # for line in lines:
    #     words = line.strip().split('\t')
    #     drug, protein, relation = words
    #     drug_protein_mat[int(drug), int(protein)] = 1
    #     count += 1
    # np.savetxt('drug_protein_mat1.txt', drug_protein_mat, fmt='%d')

    # 重新构造kg
    # file_paths = [
    #     "D:\Document\Data\drugbank_base2020-11-02\drug2target.tsv",
    #     "D:\Document\Data\drugbank_base2020-11-02\drug2enzyme.tsv",
    #     "D:\Document\Data\drugbank_base2020-11-02\drug2carrier.tsv",
    #     "D:\Document\Data\drugbank_base2020-11-02\drug2transporter.tsv"
    # ]
    # drug2id_file = open(r'D:\Document\Paper1\dataset\final_dataset\drug2id.json', 'r')
    # drug2id_dict = json.load(drug2id_file)
    # drug2id_file.close()
    # protein2id = {}
    # action2id = {}
    # f_w = open('new_kg.txt', 'w')
    # temp = []
    # action_map = {"other/unknown":"unknow","other":"unknow"}
    # for file_path in file_paths:
    #     f = open(file_path, 'r')
    #     lines = f.readlines()
    #     lines = lines[1:]
    #     f.close()
    #     for line in tqdm(lines):
    #         words = line.strip().split('\t')
    #         drugbank_id, protein_id, action = words
    #         drugdata_id = drug2id_dict.get(drugbank_id, -1)
    #         if drugdata_id != -1:
    #             proteindata_id = protein2id.get(protein_id, -1)
    #             try:
    #                 action = action_map[action]
    #             except:
    #                 pass
    #             action_id = action2id.get(action, -1)
    #             if proteindata_id == -1:
    #                 protein2id.update({protein_id:len(protein2id)})
    #             if action_id == -1:
    #                 action2id.update({action:len(action2id)+1})     # 关系从1开始

    #             proteindata_id = protein2id[protein_id]
    #             action_id = action2id[action]
    #             line_w = str(drugdata_id) + '\t' + str(proteindata_id) + '\t' + str(action_id) + '\n'
    #             if line_w not in temp:
    #                 temp.append(line_w)
    #                 f_w.write(line_w)
    # f_w.close()
    # f1 = open('protein2id_new.json', 'w')
    # f2 = open('action2id_new.json', 'w')
    # json.dump(protein2id,f1,indent=4,separators=(',',':'))
    # json.dump(action2id,f2,indent=4,separators=(',',':'))
    # f1.close()
    # f2.close()
        
    # 构造drug-indication_mat
    # f = open('indication.txt', 'r')
    # lines = f.readlines()
    # f.close()
    # drug_protein_mat = np.zeros((1177, 1546))
    # count = 1
    # temp = {}
    # for line in lines:
    #     words = line.strip().split('\t')
    #     drug, protein = words
    #     temp[protein] = temp.get(protein,len(temp))
    #     drug_protein_mat[int(drug), int(temp[protein])] = 1
    #     count += 1
    # np.savetxt('drug_indication_mat.txt', drug_protein_mat, fmt='%d')

    # 编码id2drug
    # adr2id_path = r'D:\Document\Paper1\dataset\final_dataset1\adr2id.json'
    # drug2id_path = r'D:\Document\Paper1\dataset\final_dataset1\drug2id.json'
    # f_adr = open(adr2id_path, 'r')
    # f_drug = open(drug2id_path, 'r')
    # adr2id = json.load(f_adr)
    # drug2id = json.load(f_drug)
    # f_adr.close()
    # f_drug.close()

    # f_adr = open('id2adr.json','w')
    # f_drug = open('id2drug.json','w')

    # temp = {}
    # for k, v in adr2id.items():
    #     temp[int(v)] = k
    # json.dump(temp, f_adr, indent=4,separators=(',',':'))

    # temp = {}
    # for k, v in drug2id.items():
    #     temp[int(v)] = k
    # json.dump(temp, f_drug, indent=4,separators=(',',':'))

    # f_adr.close()
    # f_drug.close()

    # 五折交叉验证切分数据集
    print('loading data')
    drug_se_data = np.loadtxt('sider.txt', dtype=int)
    kg = np.loadtxt('kg.txt', dtype=int)
    indication = np.loadtxt('indication.txt', dtype=int)

    # 编码序号，drug(0-1176)->protein(1177-2925)->adr(2926-7173)
    indication[:,1] += 2926
    x = np.zeros((indication.shape[0], 1))
    indication = np.concatenate([indication, x], axis=1)    # has_indication:0
    kg[:,1] += 1177     # kg 其他关系(1-49)
    kg = np.concatenate([kg, indication]).astype(int)

    kf = KFold(n_splits=5, shuffle=True, random_state=123)
    i = 0
    for train_idx, test_idx in kf.split(drug_se_data):
        train_idx, valid_idx = train_test_split(train_idx, test_size=0.05)
        train_set = drug_se_data[train_idx]
        valid_set = drug_se_data[valid_idx]
        test_set = drug_se_data[test_idx]

        # np.savetxt(f'{i}\\train_set.txt', train_set, fmt='%d')
        # np.savetxt(f'{i}\\valid_set.txt', valid_set, fmt='%d')
        # np.savetxt(f'{i}\\test_set.txt', test_set, fmt='%d')
        # np.savetxt(f'{i}\\kg.txt', kg, fmt='%d')

        train_set[:,2] = 50 # has_adr:50
        train_kg = np.concatenate([kg, train_set]).astype(int)
        np.savetxt(f'{i}\\train_kg.txt', train_kg, fmt='%d')
        i += 1

    
    print('finished')