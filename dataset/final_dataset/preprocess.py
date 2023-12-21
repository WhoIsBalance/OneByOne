import numpy as np
import json
from tqdm import tqdm

if __name__ == "__main__":
    # f = open('kg.txt', 'r')
    # lines = f.readlines()
    # f.close()
    # drug_protein_mat = np.zeros((1177, 1749))
    # count = 1
    # for line in lines:
    #     words = line.strip().split('\t')
    #     drug, protein, relation = words
    #     drug_protein_mat[int(drug), int(protein)-1177] = 1
    #     count += 1
    # np.savetxt('drug_protein_mat1.txt', drug_protein_mat, fmt='%d')

    # 重新构造kg
    file_paths = [
        "D:\Document\Data\drugbank_base2020-11-02\drug2target.tsv",
        "D:\Document\Data\drugbank_base2020-11-02\drug2enzyme.tsv",
        "D:\Document\Data\drugbank_base2020-11-02\drug2carrier.tsv",
        "D:\Document\Data\drugbank_base2020-11-02\drug2transporter.tsv"
    ]
    drug2id_file = open(r'D:\Document\Paper1\dataset\final_dataset\drug2id.json', 'r')
    drug2id_dict = json.load(drug2id_file)
    drug2id_file.close()
    protein2id = {}
    action2id = {}
    f_w = open('new_kg.txt', 'w')
    temp = []
    action_map = {"other/unknown":"unknow","other":"unknow"}
    for file_path in file_paths:
        f = open(file_path, 'r')
        lines = f.readlines()
        lines = lines[1:]
        f.close()
        for line in tqdm(lines):
            words = line.strip().split('\t')
            drugbank_id, protein_id, action = words
            drugdata_id = drug2id_dict.get(drugbank_id, -1)
            if drugdata_id != -1:
                proteindata_id = protein2id.get(protein_id, -1)
                try:
                    action = action_map[action]
                except:
                    pass
                action_id = action2id.get(action, -1)
                if proteindata_id == -1:
                    protein2id.update({protein_id:len(protein2id)})
                if action_id == -1:
                    action2id.update({action:len(action2id)+1})     # 关系从1开始

                proteindata_id = protein2id[protein_id]
                action_id = action2id[action]
                line_w = str(drugdata_id) + '\t' + str(proteindata_id) + '\t' + str(action_id) + '\n'
                if line_w not in temp:
                    temp.append(line_w)
                    f_w.write(line_w)
    f_w.close()
    f1 = open('protein2id_new.json', 'w')
    f2 = open('action2id_new.json', 'w')
    json.dump(protein2id,f1,indent=4,separators=(',',':'))
    json.dump(action2id,f2,indent=4,separators=(',',':'))
    f1.close()
    f2.close()
        

