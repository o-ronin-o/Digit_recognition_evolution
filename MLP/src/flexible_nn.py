# This file is MLP/src/flexible_nn.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class FlexibleNN(nn.Module):

    def __init__(self, layer_sizes):
        super().__init__()
        
        layers = []
        for i in range(len(layer_sizes) - 2):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
        
        self.network = nn.Sequential(*layers)
        
        self.network.apply(self._init_weights)
        
        # History for plotting
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    def _init_weights(self, module):
        """Applies Kaiming (He) Uniform initialization (Requirement B1)."""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        """Defines the forward pass."""
        x = x.view(x.shape[0], -1)
        return self.network(x)
        
    def predict(self, X):
        """Helper function to get predictions (class indices)."""
        self.eval()
        with torch.no_grad():
            logits = self(X)
            preds = torch.argmax(logits, dim=1)
        return preds

    def fit(self, train_loader, val_loader, epochs=50, lr=0.01):
        """
        The complete training loop.
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=lr)
        
        for epoch in range(epochs):
            self.train() 
            train_loss, train_correct, train_total = 0, 0, 0
            
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                logits = self(X_batch) 
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * X_batch.size(0)
                preds = torch.argmax(logits, dim=1)
                train_correct += (preds == y_batch).sum().item()
                train_total += y_batch.size(0)

            avg_train_loss = train_loss / train_total
            avg_train_acc = train_correct / train_total
            
            self.eval() 
            val_loss, val_correct, val_total = 0, 0, 0
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    logits = self(X_batch)
                    loss = criterion(logits, y_batch)
                    val_loss += loss.item() * X_batch.size(0)
                    preds = torch.argmax(logits, dim=1)
                    val_correct += (preds == y_batch).sum().item()
                    val_total += y_batch.size(0)
            
            avg_val_loss = val_loss / val_total
            avg_val_acc = val_correct / val_total
            
            print(f"Epoch {epoch+1:2}/{epochs} | "
                f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc*100:.2f}% | "
                f"Val Loss: {avg_val_loss:.4f},   Val Acc: {avg_val_acc*100:.2f}%")
            
            self.history['train_loss'].append(avg_train_loss)
            self.history['train_acc'].append(avg_train_acc)
            self.history['val_loss'].append(avg_val_loss)
            self.history['val_acc'].append(avg_val_acc)