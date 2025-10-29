import torch
import torch.nn.functional as F
import torch.nn as nn
class SoftmaxRegressionScratch:

    def __init__(self, input_dim, num_classes=10, lr=0.01):
        torch.manual_seed(8)
        super().__init__()
        self.lr = lr

        self.W = nn.Parameter(torch.randn(784, num_classes) * 0.01)
        self.b = nn.Parameter(torch.zeros(num_classes))


        # Tracking
        self.loss_history = []
        self.weight_norm_history = []
        self.train_acc_history = []

    def compute_loss(self, X, y):
        logits = X @ self.W + self.b  # shape: (batch, 10)

        # Softmax + cross-entropy in one function for stability
        loss = F.cross_entropy(logits, y)

        return loss, logits

    def predict(self, X):
        X = X.view(X.shape[0], -1)
        logits = X @ self.W + self.b
        preds = torch.argmax(logits, dim=1)
        return preds

    def fit(self, data_loader, epochs=10):
        for epoch in range(epochs):
            total_loss = 0
            correct, total = 0, 0

            for X, y in data_loader:
                X = X.float().view(X.shape[0], -1)

                loss, logits = self.compute_loss(X, y)

                loss.backward()

                with torch.no_grad():
                    self.W -= self.lr * self.W.grad
                    self.b -= self.lr * self.b.grad
                    self.W.grad.zero_()
                    self.b.grad.zero_()

                total_loss += loss.item() * X.shape[0]
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

                # Track metrics per batch
                self.loss_history.append(loss.item())
                self.weight_norm_history.append(torch.norm(self.W).item())

            train_acc = correct / total
            self.train_acc_history.append(train_acc)

            avg_loss = total_loss / len(data_loader.dataset)
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {train_acc*100:.2f}%")

        return avg_loss
