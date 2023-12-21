import argparse

def parameter_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_drug',type=int,default=1177,help='number of drugs')
    parser.add_argument('--n_protein',type=int,default=1749,help='number of proteins')
    parser.add_argument('--n_adr',type=int,default=4247,help='number of adrs')
    parser.add_argument('--n_node', type=int, default=7173)
    parser.add_argument('--n_relation',type=int,default=51,help='number of relations')
    parser.add_argument('--n_indication',type=int,default=1546,help='number of relations')
    parser.add_argument('--root', type=str, default='..\\..\\dataset\\final_dataset1')
    parser.add_argument('--save_path', type=str, default='..\\result')

    parser.add_argument('--u', type=float, default=2e-3)
    parser.add_argument('--v', type=float, default=2e-5)

    return parser.parse_args()