import argparse

def parameter_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--n_drug',type=int,default=1177,help='number of drugs')
    parser.add_argument('--n_protein',type=int,default=1749,help='number of proteins')
    parser.add_argument('--n_adr',type=int,default=4247,help='number of adrs')
    parser.add_argument('--n_entity',type=int,default=2926,help='number of entities')
    parser.add_argument('--n_relation',type=int,default=51,help='number of relations')
    parser.add_argument('--sider_scr',type=str,default='..\\..\\dataset\\final_dataset1\\sider.txt')
    parser.add_argument('--sider_mat',type=str,default='..\\..\\dataset\\final_dataset1\\drug_se_mat.txt')
    parser.add_argument('--save_path', type=str ,default= '..\\result')
    parser.add_argument('--kg_scr',type=str,default='..\\..\\dataset\\final_dataset1\\kg.txt')
    parser.add_argument('--protein_mat', type=str, default='..\\..\\dataset\\final_dataset1\\drug_protein_mat1.txt')
    parser.add_argument('--indication_scr', type=str, default='..\\..\\dataset\\final_dataset1\\indication.txt')
    parser.add_argument('--n_indication',type=int,default=1546,help='number of relations')
    parser.add_argument('--root', type=str, default='..\\..\\dataset\\final_dataset1')


    return parser.parse_args()