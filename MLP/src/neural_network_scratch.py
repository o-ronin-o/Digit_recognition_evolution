import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm # Optional: for a nice progress bar

class NeuralNetworkScratch(nn.Module):
    
    def __init__(self, input_dim, hidden1, hidden2, output_dim, seed=47):
        """
        architecture: Input -> Hidden1 -> Hidden2 -> Output
        """
        super().__init__()
        torch.manual_seed(seed)
        # Define the layers
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_dim)
        
        # Proper weight initialization (Xavier)
        self.apply(self._init_weights)
        
        # History for plotting
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

    def _init_weights(self, module):
        """Applies Xavier Uniform initialization to layers"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Defines the forward pass: Input -> ReLU -> Hidden1 -> ReLU -> Hidden2 -> Output
        """
        # Flatten the image
        x = x.view(x.shape[0], -1)
        
        # Pass through layers with ReLU activation
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def predict(self, X):
        """Helper function to get predictions (class indices)."""
        self.eval() # Set model to evaluation mode
        with torch.no_grad():
            logits = self(X)
            preds = torch.argmax(logits, dim=1)
        return preds

    def fit(self, train_loader, val_loader, epochs=50, lr=0.01):
        """
        Training Loop
        """
        # Loss function and Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=lr)
        
        for epoch in range(epochs):
            
            # --- Training Phase ---
            self.train() # Set model to training mode
            train_loss, train_correct, train_total = 0, 0, 0
            
            # B2: Batch processing with progress tracking
            for X_batch, y_batch in train_loader:
                
                # B2: Proper gradient computation
                optimizer.zero_grad()
                
                # Forward pass
                logits = self(X_batch)
                
                # Compute loss
                loss = criterion(logits, y_batch)
                
                # Backpropagation
                loss.backward()
                
                # Update weights
                optimizer.step()
                
                # Track metrics
                train_loss += loss.item() * X_batch.size(0)
                preds = torch.argmax(logits, dim=1)
                train_correct += (preds == y_batch).sum().item()
                train_total += y_batch.size(0)

            avg_train_loss = train_loss / train_total
            avg_train_acc = train_correct / train_total
            
            # --- Validation Phase ---
            self.eval() # Set model to evaluation mode
            val_loss, val_correct, val_total = 0, 0, 0
            
            # Validation split handling
            with torch.no_grad(): # Disable gradient calculation
                for X_batch, y_batch in val_loader:
                    
                    logits = self(X_batch)
                    
                    loss = criterion(logits, y_batch)
                    
                    val_loss += loss.item() * X_batch.size(0)
                    preds = torch.argmax(logits, dim=1)
                    val_correct += (preds == y_batch).sum().item()
                    val_total += y_batch.size(0)
            
            avg_val_loss = val_loss / val_total
            avg_val_acc = val_correct / val_total
            
            # B2: Progress logging
            print(f"Epoch {epoch+1:2}/{epochs} | "
                f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc*100:.2f}% | "
                f"Val Loss: {avg_val_loss:.4f},   Val Acc: {avg_val_acc*100:.2f}%")
            
            # Save history for plotting
            self.history['train_loss'].append(avg_train_loss)
            self.history['train_acc'].append(avg_train_acc)
            self.history['val_loss'].append(avg_val_loss)
            self.history['val_acc'].append(avg_val_acc)
            
        return self.history