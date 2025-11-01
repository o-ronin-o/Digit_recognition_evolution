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

def plot_learning_curves_with_error_bars(history_list):
    
    # Stack metrics from all runs
    # This creates a 2D numpy array: (num_runs, num_epochs)
    train_loss = np.array([h['train_loss'] for h in history_list])
    val_loss = np.array([h['val_loss'] for h in history_list])
    train_acc = np.array([h['train_acc'] for h in history_list])
    val_acc = np.array([h['val_acc'] for h in history_list])
    
    # Get mean and standard deviation across all runs (axis=0)
    mean_train_loss, std_train_loss = np.mean(train_loss, axis=0), np.std(train_loss, axis=0)
    mean_val_loss, std_val_loss = np.mean(val_loss, axis=0), np.std(val_loss, axis=0)
    mean_train_acc, std_train_acc = np.mean(train_acc, axis=0), np.std(train_acc, axis=0)
    mean_val_acc, std_val_acc = np.mean(val_acc, axis=0), np.std(val_acc, axis=0)
    
    epochs = np.arange(len(mean_train_loss))

    # Create the figure
    plt.figure(figsize=(12, 5))

    # --- Loss Plot ---
    plt.subplot(1, 2, 1)
    # Plot the mean lines
    plt.plot(epochs, mean_train_loss, 'b-', label="Mean Training Loss")
    plt.plot(epochs, mean_val_loss, 'r-', label="Mean Validation Loss")
    
    # Plot the shaded error bars (mean +/- std)
    plt.fill_between(epochs, mean_train_loss - std_train_loss, 
                mean_train_loss + std_train_loss, color='blue', alpha=0.2)
    plt.fill_between(epochs, mean_val_loss - std_val_loss, 
                mean_val_loss + std_val_loss, color='red', alpha=0.2)
    
    plt.title("Training & Validation Loss (Mean +/- Std)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    # --- Accuracy Plot ---
    plt.subplot(1, 2, 2)
    # Plot the mean lines
    plt.plot(epochs, mean_train_acc, 'b-', label="Mean Training Accuracy")
    plt.plot(epochs, mean_val_acc, 'r-', label="Mean Validation Accuracy")
    
    # Plot the shaded error bars (mean +/- std)
    plt.fill_between(epochs, mean_train_acc - std_train_acc, 
                    mean_train_acc + std_train_acc, color='blue', alpha=0.2)
    plt.fill_between(epochs, mean_val_acc - std_val_acc, 
                    mean_val_acc + std_val_acc, color='red', alpha=0.2)
    
    plt.title("Training & Validation Accuracy (Mean +/- Std)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
