import torch
import torch.nn as nn
from tqdm import tqdm

class LinearModel(nn.Module):
    
    def __init__(self, n_kernel, n_iter=100, lr=0.1) -> None:
        super(LinearModel, self).__init__()
        self.parms = nn.Parameter(torch.ones(n_kernel))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.n_iter = n_iter

    def forward(self, kernels):
        out = 0
        parms = torch.softmax(self.parms, dim=0)
        for i in range(len(kernels)):
            kernel = torch.from_numpy(kernels[i])
            out += parms[i] * kernel
        return out

    def fit(self, kernels, ideal_kernel):

        loss_list = []
        ideal_kernel = torch.from_numpy(ideal_kernel)
        for i in tqdm(range(self.n_iter), desc='training LinearModel'):
            out = self.forward(kernels)       
            x = torch.flatten(out)
            y = torch.flatten(ideal_kernel)
            loss = torch.cosine_similarity(x, y, dim=0) * -1
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_list.append(loss.cpu().detach().numpy())
        
        return loss_list
    
    def combination(self, kernels):
        parms = torch.softmax(self.parms, dim=0)
        parms = parms.detach().numpy()
        out = 0
        for i in range(len(parms)):
            out += parms[i] * kernels[i]
        
        return out
    
    def get_parms(self):
        parms = torch.softmax(self.parms, dim=0)
        parms = parms.detach().numpy()

        return parms

