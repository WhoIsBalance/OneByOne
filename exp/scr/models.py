import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from utils import *

class FBMF(nn.Module):
    '''
    full-batch training for matrix factorize
    '''
    def __init__(self, drug_embeddings:torch.Tensor, adr_embeddings:torch.Tensor, lr=0.2) -> None:
        super(FBMF, self).__init__()
        self.drug_embeddings = nn.Parameter(drug_embeddings)
        self.adr_embeddings = nn.Parameter(adr_embeddings)
        self.h = nn.Parameter(torch.empty(1, drug_embeddings.shape[1]))
        nn.init.xavier_normal_(self.h)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
    
    def forward(self):
         scores = torch.mm((self.h * self.drug_embeddings), self.adr_embeddings.T)
        # scores = self.drug_embeddings * s
         return scores
    
    def fit(self, train_mat, epochs=500, weight=1, lamb=1e-4):
        
        for e in range(epochs):
            self.scores = self.forward()
            loss = F.binary_cross_entropy_with_logits(self.scores, train_mat.float(), reduction='none')
            loss = torch.mean(loss * weight)
            reg = torch.norm(self.drug_embeddings, 2) + torch.norm(self.adr_embeddings, 2) + torch.norm(self.h, 2)
            loss = loss + lamb * reg
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def inference(self, logit=True):
        if logit == True:
            return torch.sigmoid(self.forward().detach())
        else:
            return self.forward().detach()
    

class MBMF(nn.Module):
    '''
    mini-batch training for matrix factorize
    '''
    def __init__(self, drug_embeddings:torch.Tensor, adr_embeddings:torch.Tensor) -> None:
        super(MBMF, self).__init__()
        self.drug_embeddings = nn.Parameter(drug_embeddings)
        self.adr_embeddings = nn.Parameter(adr_embeddings)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        

    def forward(self, drug_idx:torch.Tensor, adr_idx:torch.Tensor):
        drug = self.drug_embeddings[drug_idx]
        adr = self.adr_embeddings[adr_idx]
        scores = torch.sum(drug * adr, dim=1)
        return scores

    def fit(self, x:torch.Tensor, y:torch.Tensor, epochs=10):
        loader = self.mini_batch(x, y) 
        for e in tqdm(range(epochs), desc='mb-MF'):
            for step, (batch_x, batch_y) in enumerate(loader):
                drug_idx = batch_x[:,0].long()
                adr_idx = batch_x[:,1].long()
                labels = batch_y.float()
                scores = self.forward(drug_idx, adr_idx)
                loss = F.binary_cross_entropy_with_logits(scores, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def inference(self, drug_idx:torch.Tensor, adr_idx:torch.Tensor):
        with torch.no_grad():
            drug_idx = drug_idx.long()
            adr_idx = adr_idx.long()
            scores = self.forward(drug_idx, adr_idx)
        
        return drug_idx, adr_idx, torch.sigmoid(scores)

    def mini_batch(self, x:torch.Tensor, y:torch.Tensor) -> data.DataLoader:

        torch_dataset = data.TensorDataset(x, y)
        loader = data.DataLoader(
            dataset=torch_dataset,
            batch_size=2048,             # 每批提取的数量
            shuffle=True,             # 要不要打乱数据（打乱比较好）
            num_workers=2             # 多少线程来读取数据
        )
        return loader


class LogitMF(nn.Module):
    def __init__(self, fpts:torch.Tensor, dim=100, n_drug=1177, n_adr=4247) -> None:
        super(LogitMF, self).__init__()
        self.drug_embeddings = nn.Parameter(fpts.float())
        self.adr_embeddings = nn.Parameter(torch.empty(n_adr, dim))
        self.bias_d = nn.Parameter(torch.empty(n_drug, 1))
        self.bias_a = nn.Parameter(torch.empty(n_adr, 1))
        nn.init.xavier_normal_(self.adr_embeddings)
        nn.init.xavier_normal_(self.bias_d)
        nn.init.xavier_normal_(self.bias_a)
        self.L = nn.Linear(fpts.shape[1], dim)


    def forward(self, drug_idx:torch.Tensor, adr_idx:torch.Tensor):
        drug_idx = drug_idx.long()
        adr_idx = adr_idx.long()
        drugs = self.drug_embeddings[drug_idx]
        adrs = self.adr_embeddings[adr_idx]
        bias_d = self.bias_d[drug_idx]
        bias_a = self.bias_a[adr_idx]
        drugs = self.L.forward(drugs)
        scores = torch.sum(drugs * adrs, dim=1, keepdim=True) + bias_a + bias_d
        return scores.flatten()
    
    def fit(self, x, labels, lr, epochs, lamb=1e-4):
        loader = self.mini_batch(x, labels) 
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for e in tqdm(range(epochs)):
            for step, (batch_x, batch_y) in enumerate(loader):
                weight = (batch_y == 0).float() * 0.5
                weight = torch.ones_like(weight, dtype=torch.float32) + weight
                scores = self.forward(batch_x[:,0], batch_x[:,1])
                reg = torch.norm(self.drug_embeddings, 2) + torch.norm(self.adr_embeddings, 2)
                loss = F.binary_cross_entropy_with_logits(scores, batch_y, reduction='none')
                loss = torch.mean(loss * weight)
                loss = loss + reg * lamb
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


    def inference(self, drug_idx, adr_idx):
        with torch.no_grad():
            scores = self.forward(drug_idx, adr_idx)
        return drug_idx, adr_idx, torch.sigmoid(scores)


    def predition(self, scores:torch.Tensor, labels:torch.Tensor):
        def f1(p, r):
            if p + r != 0:
                return 2 * (p * r) / (p + r)
            else:
                return 0
        scores = scores.numpy()
        labels = labels.numpy()
        precisions, recalls, thresholds = precision_recall_curve(labels, scores)
        f1s=[f1(p, r) for p, r in zip(precisions, recalls)]
        optimal_index = np.argmax(f1s)
        optimal_threshold = thresholds[optimal_index]
        pred = (scores > optimal_threshold).astype(int)

        return torch.from_numpy(pred)

    def mini_batch(self, x:torch.Tensor, y:torch.Tensor) -> data.DataLoader:

        torch_dataset = data.TensorDataset(x, y)
        loader = data.DataLoader(
            dataset=torch_dataset,
            batch_size=2048,             # 每批提取的数量
            shuffle=True,             # 要不要打乱数据（打乱比较好）
            num_workers=2             # 多少线程来读取数据
        )
        return loader


class Forget():
    def __init__(self, train_mat) -> None:
        self.pre_acc = torch.zeros_like(train_mat)
        self.learning_state = torch.ones_like(train_mat)

    def count(self, train_mat, scores, thres):
        a = (scores > thres).int()  #分数大于分类平面的为1，否则为0
        cur_acc = (train_mat == a).int()    # 预测对的样本，准确率为1
        learning_state = cur_acc - self.pre_acc      #   值为1， 则是learned examples, 值为0， 则是与之前的结果相等
                                                    #   值为-1， 则是forgetable examples
        self.pre_acc = cur_acc
        forget_x, forget_y = torch.where(learning_state == -1)
        # learn_x, learn_y = torch.where(learning_state == 1)
        
        forget_target = train_mat[forget_x, forget_y]
        return torch.stack((forget_x, forget_y)).T, forget_target
    
    def get_uncetain_expamles(self, train_mat, scores, thres):
        
        mask1 = torch.zeros_like(train_mat)
        mask2 = torch.zeros_like(train_mat)
        mask1_idx = (scores >= thres).nonzero()
        mask2_idx = (scores <= thres + 0.1).nonzero()
        mask1[mask1_idx[:,0], mask1_idx[:,1]] = 1
        mask2[mask2_idx[:,0], mask2_idx[:,1]] = 1
        mask = mask1 * mask2
        forget_x, forget_y = torch.where(mask == 1)
        x = torch.stack((forget_x, forget_y)).T
        forget_target = train_mat[x[:,0], x[:,1]]
        return x, forget_target


class MutiMF():
    
    def __init__(self, drug_embedidngs, adr_embedidngs, valid_set, fpts) -> None:
        self.drug_embeddings = drug_embedidngs
        self.adr_embeddings = adr_embedidngs
        self.valid_set = valid_set
        self.fpts = fpts

    def fit(self, train_mat, epochs=20):
        fbmf = FBMF(self.drug_embeddings, self.adr_embeddings)
        mbmf = MBMF(self.drug_embeddings, self.adr_embeddings)
        for i in range(epochs):
            forgeter = Forget(train_mat)
            for e in range(1):
                fbmf.fit(train_mat, epochs=100)
                scores_mat = fbmf.inference()
                thres = find_thres(self.valid_set, scores_mat)
                x, y = forgeter.get_uncetain_expamles(train_mat, scores_mat.detach(), thres)
                if len(y) > 0:
                    mbmf.fit(x, y, epochs=3)
                    drug_idx = x[:,0].long()
                    adr_idx = x[:,1].long()
                    drug_idx, adr_idx, scores_fix = mbmf.inference(drug_idx, adr_idx)
                scores_mat[drug_idx, adr_idx] = scores_fix
                scores = scores_mat.detach().numpy()
                thres = find_thres(self.valid_set, scores)
                aupr, f1, prec, recall, mcc, mr, scores, scores_ = eval(self.valid_set, scores,thres)
                
                print(f'aupr:{aupr}    f1:{f1}')
        self.scores_mat = scores_mat
        return scores_mat
    
    def inference(self):

        return self.scores_mat.detach().numpy()
    
    def predition(self, drug_idx, adr_idx, scores_mat, pred_fix, train_mat) -> np.ndarray:
        '''
        二次分类用的drug_idx和adr_idx,  thres1 是二次分类用的阈值,  thres2则是正常阈值
        '''
        def f1(p, r):
            if p + r != 0:
                return 2 * (p * r) / (p + r)
            else:
                return 0
        scores_mat[drug_idx, adr_idx] = -1
        x, y = torch.where(scores_mat > -1)
        scores = scores_mat[x, y]
        labels = train_mat[x, y]
        scores = scores.numpy()
        labels = labels.numpy()
        precisions, recalls, thresholds = precision_recall_curve(labels, scores)
        f1s=[f1(p, r) for p, r in zip(precisions, recalls)]
        optimal_index = np.argmax(f1s)
        optimal_threshold = thresholds[optimal_index]
        pred = (scores > optimal_threshold).astype(int)
        result = np.zeros_like(train_mat)
        result[x, y] = pred
        result[drug_idx.numpy(), adr_idx.numpy()] = pred_fix

        return result
    
    def eval(self, result, test_set):

        drug = test_set[:,0]
        adr = test_set[:,1]
        labels = test_set[:,2]

        pred = result[drug, adr]
        from sklearn.metrics import f1_score
        f1 = f1_score(labels, pred)

        return f1
    

class LTMF(nn.Module):

    def __init__(self, drug_embeddings, adr_embeddings, dim) -> None:
        super(LTMF, self).__init__()
        self.drug_embeddings = nn.Parameter(drug_embeddings)
        self.adr_embeddings = nn.Parameter(adr_embeddings)
        self.dim = dim

    def forward(self):
        drug_freq_embeddings = self.drug_freq_onehot.mm(self.drugfreq_embeddings)
        adr_freq_embeddings = self.adr_freq_onehot.mm(self.adrfreq_embeddings)
        drug_embeddings = drug_freq_embeddings*0.1 + self.drug_embeddings
        adr_embeddings = adr_freq_embeddings*0.1 + self.adr_embeddings
        scores = torch.mm(drug_embeddings, adr_embeddings.T)

        return scores

    def fit(self, train_mat:torch.Tensor, epochs=500, lr=0.2):
        drug_freq = torch.sum(train_mat, dim=1)
        adr_freq = torch.sum(train_mat, dim=0)
        self.drug_freq_onehot = self.freqencode(drug_freq).float()
        self.adr_freq_onehot = self.freqencode(adr_freq).float()
        self.drugfreq_embeddings = nn.Parameter(torch.empty(self.drug_freq_onehot.shape[1], self.dim))
        self.adrfreq_embeddings = nn.Parameter(torch.empty(self.adr_freq_onehot.shape[1], self.dim))
        nn.init.xavier_normal_(self.drugfreq_embeddings)
        nn.init.xavier_normal_(self.adrfreq_embeddings)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for e in range(epochs):
            self.scores = self.forward()
            loss = F.binary_cross_entropy_with_logits(self.scores, train_mat.float())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
    def inference(self, logit=True):
        if logit == True:
            return torch.sigmoid(self.forward().detach())
        else:
            return self.forward().detach()

    def freqencode(self, freq:torch.Tensor):
        table = {}
        tmp = []
        for i in range(len(freq)):
            table.update({freq[i].item():table.get(freq[i], len(table))})
            tmp.append(table[freq[i].item()])
        onehot_code = F.one_hot(torch.tensor(tmp))     # n * n_unique
        return onehot_code