import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset,Dataset
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import torch
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error, mean_absolute_error
import pandas as pd

# we need to input :
#           - the path where the data goes 
#           - tthe validation porpotion i want relative to the whole dataset
# my output will be: 
#           - dataloader for each portion 
#           - the size of my dataset


def load_transform_split_mnist(
    val_size=0.2,
    path="../data",
    normalize=False,
    classes=None,  # None → all classes
    batch_size=64
):
    transform = transforms.Compose(
        [transforms.ToTensor()] +
        ([transforms.Normalize((0.5,), (0.5,))] if normalize else [])
    )

    full_train = datasets.MNIST(root=path, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=path, train=False, download=True, transform=transform)

    if classes is None:
        classes = list(range(10))

    train_mask = torch.isin(full_train.targets, torch.tensor(classes))
    train_indices = torch.where(train_mask)[0].tolist()

    test_mask = torch.isin(test_set.targets, torch.tensor(classes))
    test_indices = torch.where(test_mask)[0].tolist()

    class_to_idx = {c: i for i, c in enumerate(classes)}

    class CustomSubset(Dataset):
        def __init__(self, ds, indices):
            self.ds = ds
            self.indices = indices
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, i):
            x, y = self.ds[self.indices[i]]
            return x, class_to_idx[int(y)]

    stratify_labels = [class_to_idx[int(full_train.targets[i])] for i in train_indices]
    train_idx, val_idx = train_test_split(
        train_indices,
        test_size=val_size,
        stratify=stratify_labels,
        random_state=42
    )

    train_set = CustomSubset(full_train, train_idx)
    val_set = CustomSubset(full_train, val_idx)
    test_set = CustomSubset(test_set, test_indices)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, len(train_set), len(val_set), len(test_set)

def check_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon MPS GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU (no GPU available)")
    
    print("Device:", device)

    torch.manual_seed(int(time.time()))
    return device


def count_classes(data_loader):
    labels = []
    for _, y in data_loader:  # iterate over batches
        labels.append(y)
    labels = torch.cat(labels)
    counts = Counter(labels.cpu().numpy())
    return counts



def classification_metrics(y_true, y_pred):
   
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred,average="binary"),
        "recall": recall_score(y_true, y_pred,average="binary"),
        "f1_score": f1_score(y_true, y_pred,average="binary"),
        "confusion_matrix": confusion_matrix(y_true, y_pred)
    }


def evaluate_softmax(model, data_loader):
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for X, y in data_loader:
            X = X.float().view(X.shape[0], -1)

            loss, logits = model.compute_loss(X, y)
            total_loss += loss.item() * X.shape[0]

            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc



def per_class_accuracy(cm):
    correct = np.diag(cm)
    total = cm.sum(axis=1)
    acc_per_class = correct / total
    
    df = pd.DataFrame({
        "Class": list(range(10)),
        "Accuracy": acc_per_class
    })

    print(df)
    return df
