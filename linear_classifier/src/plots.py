import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

def show_samples(data_loader):
    images,labels = next(iter(data_loader))

    plt.figure(figsize=(8,8))
    for i in range(9):
        plt.subplot(3,3, i+1)
        plt.imshow(images[i].squeeze(), cmap = 'gray')
        plt.title(f"Label: {labels[i].item()}")
        plt.axis('off')
    plt.show()

