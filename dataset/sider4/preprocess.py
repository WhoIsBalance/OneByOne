import numpy as np

if __name__ == "__main__":
    a = np.loadtxt('drug_protein_mat.txt')
    a[a > 0] = 1
    np.savetxt('drug_protein_mat1.txt', a, fmt='%d')