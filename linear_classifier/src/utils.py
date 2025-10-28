import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import torch
import time

# we need to input :
#           - the path where the data goes 
#           - tthe validation porpotion i want relative to the whole dataset
# my output will be: 
#           - dataloader for each portion 
#           - the size of my dataset

def load_transform_split_mnist(val_size = .2, path = "../data", normalize = False):
    transform_list = [transforms.ToTensor()]
    if normalize :
        transform_list.append(transforms.Normalize((.5,),(.5,)))

    transform = transforms.Compose(transform_list)
    # Load the original MNIST train and test sets
    full_train = datasets.MNIST(root=path, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=path, train=False, download=True, transform=transform)

    # Creating split: 60% train, 20% validation, 20% test
    train_indices, val_indices = train_test_split(
        list(range(len(full_train))),
        test_size=val_size,
        stratify=full_train.targets,
        random_state=42
    )

    train_set = Subset(full_train, train_indices)
    val_set = Subset(full_train, val_indices)

    # Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
    return train_loader,val_loader,test_loader , len(train_set),len(val_set), len(test_set)



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
    images, labels = next(iter(data_loader))
    targets = data_loader.dataset.dataset.targets[data_loader.dataset.indices]

    counts = Counter(targets.cpu().numpy())
    return counts


