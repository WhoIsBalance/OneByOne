import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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


    def forward(self):
        drugs = self.drug_embeddings
        adrs = self.adr_embeddings
        bias_d = self.bias_d
        bias_a = self.bias_a
        drugs = self.L.forward(drugs)
        scores = torch.mm(drugs, adrs.T) + bias_a.T + bias_d
        return scores
    
    def fit(self, labels, lr, epochs, lamb=1e-4):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for e in range(epochs):
            scores = self.forward()
            reg = torch.norm(self.drug_embeddings, 2) + torch.norm(self.adr_embeddings, 2)
            loss = F.binary_cross_entropy_with_logits(scores, labels.float())
            loss = loss + reg * lamb
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def inference(self):
        with torch.no_grad():
            scores = self.forward()
        
        return scores.detach()