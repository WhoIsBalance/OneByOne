import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from param_parser import parameter_parser
import utils
import torch.nn as nn

args = parameter_parser()

class FGRMF:

    def __init__(self, train_mat, fe) -> None:

        self.dim_k = args.dim_k
        self.mu = args.mu
        self.lambda_ = args.lambda_
        self.threshold = 0.5
        self.n_drug = args.n_drug
        self.n_adr = args.n_adr
        self.association_matrix = train_mat
        if fe == 'chem':
            _,self.fps,self.dim_fp = utils.load_structure()
        elif fe == 'protein':
            self.fps,self.dim_fp = utils.load_protein()
        else:
            self.fps,self.dim_fp = utils.load_indication()
        self.similar_matirx = self.__init_similar_matrix()
        self.create_hidden_matrix()


    def __init_similar_matrix(self):

        similar_matirx = np.zeros((self.n_drug,self.n_drug))
        if isinstance(self.fps, list):
            fps = np.array(self.fps)
        else:
            fps = self.fps
        sum_fps = np.sum(fps,axis=1)

        for i in range(fps.shape[0]):
            same_seg_count = np.sum((fps[i] + fps) == 2,axis=1) * 2
            total_seg_count = np.sum(sum_fps[i]) + sum_fps
            w = same_seg_count / total_seg_count
            similar_matirx[i] = w
        similar_matirx[np.isnan(similar_matirx)] = 0
        
        return similar_matirx

    def create_hidden_matrix(self):

        self.hidden_matrix_x = np.random.rand(self.n_drug,self.dim_k)
        self.hidden_matrix_y = np.random.rand(self.n_adr,self.dim_k)

    def predict(self):

        self.predict_matrix = self.hidden_matrix_x.dot(self.hidden_matrix_y.T)
        return self.predict_matrix

    def train(self,keeploss=False):

        # hidden_matrix_x_ = np.zeros_like(self.hidden_matrix_x)
        # hidden_matrix_y_ = np.zeros_like(self.hidden_matrix_y)
        losses = []
        for iter_num in tqdm(range(1,args.iter+1)):
            for i in range(self.n_drug):
                # self.hidden_matrix_x[i] = self.hidden_matrix_x[i] - self.derivatives('x',i)
                self.hidden_matrix_x[i] = self.derivatives('x',i)

            for j in range(self.n_adr):
                # self.hidden_matrix_y[j] = self.hidden_matrix_y[j] - self.derivatives('y',j)
                self.hidden_matrix_y[j] = self.derivatives('y',j)
            # self.hidden_matrix_x = hidden_matrix_x_
            # self.hidden_matrix_y = hidden_matrix_y_
            if keeploss == True:
                predict_matrix = self.predict()
                loss = self.cal_loss(predict_matrix)
                losses.append(loss)
        
        return losses

    def derivatives(self,variable,i):

        if variable == 'x':
            
            de = self.association_matrix[i,:].dot(self.hidden_matrix_y)
            total = 0
            for j in range(self.n_drug):
                total += ((self.similar_matirx[i,j] + self.similar_matirx[j,i]) * self.hidden_matrix_x[j])
            de = de + total * self.lambda_

            term2 = self.hidden_matrix_y.T.dot(self.hidden_matrix_y)
            term2 += np.eye(term2.shape[0],term2.shape[1]) * self.mu
            term2 += (np.sum(self.similar_matirx[i,:]) + np.sum(self.similar_matirx[:,i])) * \
                np.eye(term2.shape[0],term2.shape[1]) * self.lambda_
            term2 = np.linalg.inv(term2)

            de = de.dot(term2)

        if variable == 'y':
            
            de = self.association_matrix[:,i].T.dot(self.hidden_matrix_x)
            term2 = self.hidden_matrix_x.T.dot(self.hidden_matrix_x)
            term2 += (np.eye(term2.shape[0],term2.shape[1]) * self.mu)
            term2 = np.linalg.inv(term2)
            de = de.dot(term2)

        return de

    def cal_loss(self,predict_matrix):

        term1 = np.sum(np.square(self.association_matrix - predict_matrix))
        # term2 = np.sum((np.sqrt(np.sum(np.square(self.hidden_matrix_x),axis=1))) + \
        #     np.sum(np.sqrt(np.sum(np.square(self.hidden_matrix_y),axis=1)))) * self.mu
        term2 = (np.linalg.norm(self.hidden_matrix_x) + np.linalg.norm(self.hidden_matrix_y)) * self.mu
        term3 = 0
        for i in range(self.hidden_matrix_x.shape[0]):
            xi = self.hidden_matrix_x[i]
            term3 += np.sum(np.sqrt(np.sum(np.square(xi - self.hidden_matrix_x),axis=1)) * \
                self.similar_matirx[i,:]) * self.lambda_

        loss = term1 + term2 + term3

        return loss

    def save(self, path_x, path_y):

        np.savetxt(path_x,self.hidden_matrix_x)
        np.savetxt(path_y,self.hidden_matrix_y)

    def load(self, path_x, path_y):

        self.hidden_matrix_x = np.loadtxt(path_x)
        self.hidden_matrix_y = np.loadtxt(path_y)


class IFGRMF(nn.Module):

    def __init__(self, train_mat, fes=['protein', 'chem', 'indication']) -> None:
        super(IFGRMF, self).__init__()
        self.models = []
        self.preds = []
        self.loss = []
        self.train_mat = train_mat
        self.fes = fes
        for fe in fes:
            self.models.append(FGRMF(train_mat, fe))
        self.L = nn.Linear(len(fes),1)
        self.optimizer = torch.optim.Adam(self.L.parameters(), lr=args.lr)
        self.threshold = 0.5
    
    def train(self, time, keeploss=True):
        
        for i in range(len(self.models)):
            sign = self.fes[i]
            x_path = args.model_save_path + f'{time}hidden_x{sign}.txt'
            y_path = args.model_save_path + f'{time}hidden_y{sign}.txt'
            if args.train_FGRMF == True:
                model = self.models[i]
                self.loss.append(model.train(keeploss=keeploss))
                self.preds.append(model.predict().flatten())
                model.save(x_path, y_path)
            else:
                model = self.models[i]
                model.hidden_matrix_x = np.loadtxt(x_path)
                model.hidden_matrix_y = np.loadtxt(y_path)
                self.preds.append(model.predict().flatten())

        if len(self.fes) > 1:
            inputs = np.stack(self.preds).T
            labels = self.train_mat.flatten()

            loss_ = []
            batches = utils.mini_batches(inputs, labels, 2048)
            for epoch in range(1, args.epochs+1):
                for batch in tqdm(batches, desc=f'integrate{epoch}'):
                    label = torch.from_numpy(batch[1]).float()
                    batch = torch.from_numpy(batch[0]).float()
                    out = self.L.forward(batch).flatten()
                    loss = torch.mean(torch.binary_cross_entropy_with_logits(out, label))

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    loss_.append(loss.detach().cpu().numpy())
            if keeploss == True:
                self.loss.append(loss_)

        return self.loss
    
    def predict(self):
        scores = []
        for pred in self.preds:
            pred_mat = pred.reshape((args.n_drug, args.n_adr))
            # scores.append(pred_mat[drug_idx, adr_idx])
            scores.append(pred_mat)
        scores = np.stack(scores).T
        if len(self.fes) > 1:
            inputs = torch.from_numpy(scores).float()
            scores = torch.sigmoid(self.L.forward(inputs))[:,:,0]
            scores = scores.detach().cpu().numpy()

        return scores.T
    
    
# class FGRMF:

#     def __init__(self,args, train_set=None, Strcture_loader=None) -> None:

#         self.args = args
#         self.dim_k = args.dim_k
#         self.mu = args.mu
#         self.lambda_ = args.lambda_
#         self.threshold = 0.5
#         self.n_adr = args.n_adr
#         self.n_drug = args.n_drug
#         if train_set is not None:
#             self.train_sider_np = train_set
#             self.association_matrix = self.__init_association_matrix(self.train_sider_np)

#         if Strcture_loader is not None:
#             self.dim_fp = Strcture_loader.dim_fp
#             self.fps = Strcture_loader.fps
#             self.similar_matirx = self.__init_similar_matrix()
#         self.create_hidden_matrix()

#     def __init_association_matrix(self,train_sider_np):

#         association_matrix = np.zeros((self.n_drug,self.n_adr))
#         for i in range(train_sider_np.shape[0]):
#             row_index,col_index,score = train_sider_np[i,0],train_sider_np[i,1],train_sider_np[i,2]
#             association_matrix[row_index,col_index] = score
        
#         return association_matrix

#     def __init_similar_matrix(self):

#         similar_matirx = np.zeros((self.n_drug,self.n_drug))
#         fps = np.array(self.fps)
#         sum_fps = np.sum(fps,axis=1)

#         for i in range(fps.shape[0]):
#             same_seg_count = np.sum((fps[i] + fps) == 2,axis=1) * 2
#             total_seg_count = np.sum(sum_fps[i]) + sum_fps
#             w = same_seg_count / total_seg_count
#             similar_matirx[i] = w
#         similar_matirx[np.isnan(similar_matirx)] = 0
        
#         return similar_matirx

#     def create_hidden_matrix(self):

#         self.hidden_matrix_x = np.random.rand(self.n_drug,self.dim_k)
#         self.hidden_matrix_y = np.random.rand(self.n_adr,self.dim_k)

#     def predict(self):

#         predict_matrix = self.hidden_matrix_x.dot(self.hidden_matrix_y.T)
#         return predict_matrix

#     def train(self,show_loss=False):

#         losses = []
#         for iter_num in tqdm(range(1,self.args.iter+1)):
#             for i in range(self.n_drug):
#                 # self.hidden_matrix_x[i] = self.hidden_matrix_x[i] - self.derivatives('x',i)
#                 self.hidden_matrix_x[i] = self.derivatives('x',i)

#             for j in range(self.n_adr):
#                 # self.hidden_matrix_y[j] = self.hidden_matrix_y[j] - self.derivatives('y',j)
#                 self.hidden_matrix_y[j] = self.derivatives('y',j)
#             if show_loss == True:
#                 predict_matrix = self.predict()
#                 loss = self.cal_loss(predict_matrix)
#                 losses.append(loss)
        
#         if show_loss == True:
#             x = [i for i in range(len(losses))]
#             plt.figure()
#             plt.plot(x,losses)
#             plt.show()

#     def derivatives(self,variable,i):

#         if variable == 'x':
            
#             de = self.association_matrix[i,:].dot(self.hidden_matrix_y)
#             total = 0
#             for j in range(self.n_drug):
#                 total += ((self.similar_matirx[i,j] + self.similar_matirx[j,i]) * self.hidden_matrix_x[j])
#             de = de + total * self.lambda_

#             term2 = self.hidden_matrix_y.T.dot(self.hidden_matrix_y)
#             term2 += np.eye(term2.shape[0],term2.shape[1]) * self.mu
#             term2 += (np.sum(self.similar_matirx[i,:]) + np.sum(self.similar_matirx[:,i])) * \
#                 np.eye(term2.shape[0],term2.shape[1]) * self.lambda_
#             term2 = np.linalg.inv(term2)

#             de = de.dot(term2)

#         if variable == 'y':
            
#             de = self.association_matrix[:,i].T.dot(self.hidden_matrix_x)
#             term2 = self.hidden_matrix_x.T.dot(self.hidden_matrix_x)
#             term2 += (np.eye(term2.shape[0],term2.shape[1]) * self.mu)
#             term2 = np.linalg.inv(term2)
#             de = de.dot(term2)

#         return de

#     def cal_loss(self,predict_matrix):

#         term1 = np.sum(np.square(self.association_matrix - predict_matrix))
#         term2 = np.sum((np.sqrt(np.sum(np.square(self.hidden_matrix_x),axis=1))) + \
#             np.sum(np.sqrt(np.sum(np.square(self.hidden_matrix_y),axis=1)))) * self.mu
#         term3 = 0
#         for i in range(self.hidden_matrix_x.shape[0]):
#             xi = self.hidden_matrix_x[i]
#             term3 += np.sum(np.sqrt(np.sum(np.square(xi - self.hidden_matrix_x),axis=1)) * \
#                 self.similar_matirx[i,:]) * self.lambda_

#         loss = term1 + term2 + term3

#         return loss

#     def save(self):

#         np.savetxt(self.args.hidden_x_path,self.hidden_matrix_x)
#         np.savetxt(self.args.hidden_y_path,self.hidden_matrix_y)

#     def load(self):

#         self.hidden_matrix_x = np.loadtxt(self.args.hidden_x_path)
#         self.hidden_matrix_y = np.loadtxt(self.args.hidden_y_path)