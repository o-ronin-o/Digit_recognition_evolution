import torch
import math 
class LogisticRegressionScratch:
    
    
    def __init__(self, input_dim, num_classes,lr):
        torch.manual_seed(8)
        self.W = torch.randn(input_dim,1, requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)
        self.lr=lr
        

    @staticmethod
    def __sigmoid(z):
        return 1 / (1 + torch.exp(-z))
    

    @staticmethod
    def __cross_entropy_loss(probs,y):
        return -torch.mean(y*torch.log(probs +1e-8)+ ( 1- y)* torch.log(1 - probs + 1e-8))
    

    def compute_loss(self, X,y):
        logits = X @ self.W + self.b
        probs = self.__sigmoid(logits)
        y = y.view(-1,1).float()
        loss = self.__cross_entropy_loss(probs,y)
        return loss
    

        

    def fit(self, data_loader, epochs=10):
        return
    

    def predict(self,X):
        return