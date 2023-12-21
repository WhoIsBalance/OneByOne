from rdkit import Chem
import numpy as np
from rdkit import DataStructs
from tqdm import tqdm

f = open(r'D:\Document\Paper4\dataset\final_dataset1\structure.txt', 'r')
lines = f.readlines()
f.close()

drugs = []
for line in lines:
    pieces = line.strip().split('\t')
    smi = pieces[1]
    m = Chem.MolFromSmiles(smi)
    fps = Chem.RDKFingerprint(m)
    drugs.append(fps)

sim_mat = np.zeros((1177, 1177))
for i in tqdm(range(len(drugs))):
    for j in range(i,len(drugs)):
        a = drugs[i]
        b = drugs[j]
        sim = DataStructs.FingerprintSimilarity(a, b)
        sim_mat[i,j] = sim
        sim_mat[j,i] = sim

np.savetxt('drug_sim_mat.txt', sim_mat)