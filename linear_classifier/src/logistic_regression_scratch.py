# import torch

# class LogisticRegressionScratch:
    
    
#     def __init__(self, input_dim,lr = .001):
#         torch.manual_seed(8)
#         self.W = 1e-4 * torch.randn(input_dim,1)
#         self.W.requires_grad_(True)  
#         self.b = torch.zeros(1, requires_grad=True)
#         self.lr=lr
        

#     @staticmethod
#     def __sigmoid(z):
#         return 1 / (1 + torch.exp(-z))
    

#     @staticmethod
#     def __cross_entropy_loss(probs,y):
#         return -torch.mean(y*torch.log(probs +1e-8)+ ( 1- y)* torch.log(1 - probs + 1e-8))
    

#     def compute_loss(self, X,y):
#         logits = X @ self.W + self.b
#         #print("logits requires_grad:", logits.requires_grad)
#         probs = torch.clamp(self.__sigmoid(logits), 1e-8, 1-1e-8)

        
#         #print("probs requires_grad:", probs.requires_grad)
#         y = y.view(-1,1).float()
#         loss = self.__cross_entropy_loss(probs,y)
      
#         return loss
    

        

#     def fit(self, data_loader, epochs=10):
#         #print("meow") debugging
#         for epoch in range(epochs):

#             total_loss = 0.0
#             for X, y in data_loader:
#                 X = X.float().view(X.shape[0], -1)  # flattening images
                
#                 loss = self.compute_loss(X, y)
               
#                 loss.backward()
               
#                 with torch.no_grad():
#                     self.W -= self.lr * self.W.grad
#                     self.b -= self.lr * self.b.grad
#                     self.W.grad.zero_()
#                     self.b.grad.zero_()
#                 total_loss += loss.item() * X.shape[0]
#             print("#"*50)
#             avg_loss = total_loss / len(data_loader.dataset)
#             print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

#     def predict(self,X):
#         X = X.view(X.shape[0], -1)
#         logits = X @ self.W + self.b
#         probs = torch.sigmoid(logits)
#         return (probs >= 0.5).long().view(-1)
    

import torch

class LogisticRegressionScratch:
    
    def __init__(self, input_dim, lr=0.001):
        torch.manual_seed(8)
        self.W = 1e-4 * torch.randn(input_dim, 1)
        self.W.requires_grad_(True)  
        self.b = torch.zeros(1, requires_grad=True)
        self.lr = lr
        
    @staticmethod
    def __sigmoid(z):
        return 1 / (1 + torch.exp(-z))
    
    @staticmethod
    def __binary_cross_entropy(probs, y):
        # More numerically stable implementation
        y = y.float().view(-1, 1)
        loss = - (y * torch.log(probs + 1e-8) + (1 - y) * torch.log(1 - probs + 1e-8))
        return torch.mean(loss)
    
    def compute_loss(self, X, y):
        logits = X @ self.W + self.b
        probs = self.__sigmoid(logits)
        probs = torch.clamp(probs, 1e-8, 1-1e-8)  # Clamp after sigmoid, not before
        
        loss = self.__binary_cross_entropy(probs, y)
        return loss
    
    def fit(self, data_loader, epochs=10):
        for epoch in range(epochs):
            total_loss = 0.0
            for X, y in data_loader:
                X = X.float().view(X.shape[0], -1)  # flattening images
                
                loss = self.compute_loss(X, y)
                
                loss.backward()
                
                with torch.no_grad():
                    self.W -= self.lr * self.W.grad
                    self.b -= self.lr * self.b.grad
                    self.W.grad.zero_()
                    self.b.grad.zero_()
                
                total_loss += loss.item() * X.shape[0]
            
            avg_loss = total_loss / len(data_loader.dataset)
            print("#" * 50)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            print("meow")

    def predict(self, X):
        X = X.view(X.shape[0], -1)
        logits = X @ self.W + self.b
        probs = self.__sigmoid(logits)
        return (probs >= 0.5).long().view(-1)
