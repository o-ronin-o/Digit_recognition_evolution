import torch
import numpy as np
class LogisticRegressionScratch:
    
    
    def __init__(self, input_dim,lr = .01):
        torch.manual_seed(8)
        self.W = 1e-4 * torch.randn(input_dim,1)
        self.W.requires_grad_(True)  
        self.b = torch.zeros(1, requires_grad=True)
        self.lr=lr
        

    @staticmethod
    def __sigmoid(z):
        return 1 / (1 + torch.exp(-z))
    

    @staticmethod
    def __cross_entropy_loss(y_pred, y_true):
        epsilon = 1e-8
        y_pred = torch.clamp(y_pred, epsilon, 1.0 - epsilon)
        loss = -torch.sum(y_true * torch.log(y_pred) + (1.0 - y_true) * torch.log(1.0 - y_pred))
        return loss / y_true.shape[0]

    def compute_loss(self, X,y):
        logits = X @ self.W + self.b
        #print("logits requires_grad:", logits.requires_grad)
        probs = self.__sigmoid(logits)
         
                
        #print("probs requires_grad:", probs.requires_grad)
        y = y.view(-1,1).float()
        loss = self.__cross_entropy_loss(probs,y)
        
      
        return loss , probs
    

        

    def fit(self, data_loader, epochs=3):
        #print("meow") debugging
        self.loss_history = []
        self.weight_norm_history = []
        self.train_acc_history = []
        
        for epoch in range(epochs):
            correct = 0
            total = 0
            total_loss = 0.0
            for X, y in data_loader:
                X = X.float().view(X.shape[0], -1)  # flattening images
                
                loss, probs= self.compute_loss(X, y)
                

                if loss.item() < 0:
                    print(f"Early stopping: Loss became negative at epoch {epoch+1}")
                    return
                
                loss.backward()
               
                with torch.no_grad():
                    self.W -= self.lr * self.W.grad
                    self.b -= self.lr * self.b.grad
                    self.W.grad.zero_()
                    self.b.grad.zero_()
                total_loss += loss.item() * X.shape[0]

                preds = (probs >= 0.5).long().view(-1)
                correct += (preds == y).sum().item()
                total += y.size(0)
                self.loss_history.append(loss.item())
                self.weight_norm_history.append(torch.norm(self.W).item())

            
            train_acc = correct / total
            self.train_acc_history.append(train_acc)
            
            print("#"*50)
            avg_loss = total_loss / len(data_loader.dataset)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        return avg_loss 

    def predict(self,X):
        X = X.float().view(X.shape[0], -1)
        logits = X @ self.W + self.b
        probs = torch.sigmoid(logits)
        return (probs >= 0.5).long().view(-1)
    


    def evaluate(self, data_loader, device="cpu"):
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for X, y in data_loader:
                X = X.to(device).float().view(X.shape[0], -1)
                y = y.to(device).float()

                loss, probs = self.compute_loss(X, y)
                total_loss += loss.item() * y.size(0)

                preds = (probs >= 0.5).long().view(-1)
                correct += (preds == y.long()).sum().item()
                total += y.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy
