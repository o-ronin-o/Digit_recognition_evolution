import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns


def show_samples(data_loader):
    images,labels = next(iter(data_loader))

    plt.figure(figsize=(8,8))
    for i in range(9):
        plt.subplot(3,3, i+1)
        plt.imshow(images[i].squeeze(), cmap = 'gray')
        plt.title(f"Label: {labels[i].item()}")
        plt.axis('off')
    plt.show()

def plot_loss_and_weights(model):
    plt.figure(figsize=(10, 5))

    # Loss subplot
    plt.subplot(1, 3, 1)
    plt.plot(model.loss_history, label="Loss")
    plt.xlabel("Iterations (batches)")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)

    # Weight magnitude subplot
    plt.subplot(1, 3, 2)
    plt.plot(model.weight_norm_history, label="‖Weights‖")
    plt.xlabel("Iterations (batches)")
    plt.ylabel("Weight")
    plt.title("Weight Growth Over Training")
    plt.grid(True)


    # Training accuracy subplot
    plt.subplot(1, 3, 3)
    plt.plot(model.train_acc_history)
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    import matplotlib.pyplot as plt

def plot_loss_and_accuracy(model):
    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(model.loss_history)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(model.train_acc_history)
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_train_val_curves(history):
    """
    B3: Plots training and validation loss/accuracy curves from a history dict.
    """
    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label="Training Loss")
    plt.plot(history['val_loss'], label="Validation Loss")
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label="Training Accuracy")
    plt.plot(history['val_acc'], label="Validation Accuracy")
    plt.title("Training & Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes=[str(i) for i in range(10)]):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")
    plt.show()
    return cm

def plot_per_class_acc(df):
    plt.figure(figsize=(8, 4))
    plt.bar(df["Class"], df["Accuracy"])
    plt.xlabel("Digit Class")
    plt.ylabel("Accuracy")
    plt.title("Per-Class Accuracy (%)")
    plt.show()
