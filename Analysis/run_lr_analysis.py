import os
import sys
import torch
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))  
project_root = os.path.dirname(current_dir)             
mlp_src_path = os.path.join(project_root, 'MLP', 'src')
sys.path.append(mlp_src_path)

from utils import load_transform_split_mnist
from neural_network_scratch import NeuralNetworkScratch 

def run_lr_analysis():
    print("--- C1.1: Starting Learning Rate Analysis ---")
    
    INPUT_DIM = 784
    HIDDEN1 = 128  
    HIDDEN2 = 64   
    OUTPUT_DIM = 10
    EPOCHS = 20 
    
    learning_rates = [0.001, 0.01, 0.1, 1.0]
    lr_histories = {} 
    
    train_loader, val_loader, _, _, _, _ = load_transform_split_mnist(
        classes=None, 
        batch_size=64
    )
    
    for lr in learning_rates:
        print(f"\n--- Testing Learning Rate: {lr} ---")
        
        model = NeuralNetworkScratch(INPUT_DIM, HIDDEN1, HIDDEN2, OUTPUT_DIM)
        
        model.fit(train_loader, val_loader, epochs=EPOCHS, lr=lr)
        
        lr_histories[lr] = model.history

    print("\n--- Learning Rate Analysis Complete! ---")

    plots_dir = os.path.join(current_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True) 
    save_path = os.path.join(plots_dir, "C1_lr_analysis_curves.png")

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    for lr, history in lr_histories.items():
        if not any(torch.isnan(torch.tensor(history['val_loss']))):
            plt.plot(history['val_loss'], label=f"LR = {lr}")
    plt.title("Validation Loss vs. Learning Rate")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.0) 

    plt.subplot(1, 2, 2)
    for lr, history in lr_histories.items():
        if not any(torch.isnan(torch.tensor(history['val_loss']))):
            plt.plot(history['val_acc'], label=f"LR = {lr}")
    plt.title("Validation Accuracy vs. Learning Rate")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.ylim(0.8, 1.0) 

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Learning Rate analysis plot saved to: {save_path}")


if __name__ == "__main__":
    run_lr_analysis()