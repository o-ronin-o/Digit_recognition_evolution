import os
import sys
import torch
import matplotlib.pyplot as plt
import time
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
mlp_src_path = os.path.join(project_root, 'MLP', 'src')
sys.path.append(mlp_src_path)

from utils import load_transform_split_mnist
from neural_network_scratch import NeuralNetworkScratch

def run_bs_analysis():
    print("--- C1.2: Starting Batch Size Analysis ---")
    
    INPUT_DIM = 784
    HIDDEN1 = 128
    HIDDEN2 = 64
    OUTPUT_DIM = 10
    EPOCHS = 20
    LEARNING_RATE = 0.01 

    batch_sizes = [16, 32, 64, 128]
    bs_histories = {}
    results_list = []

    for bs in batch_sizes:
        print(f"\n--- Testing Batch Size: {bs} ---")
        
        train_loader_bs, val_loader_bs, _, _, _, _ = load_transform_split_mnist(
            classes=None, 
            batch_size=bs
        )
        
        model = NeuralNetworkScratch(INPUT_DIM, HIDDEN1, HIDDEN2, OUTPUT_DIM)
        
        start_time = time.time()
        model.fit(train_loader_bs, val_loader_bs, epochs=EPOCHS, lr=LEARNING_RATE)
        end_time = time.time()
        
        bs_histories[bs] = model.history
        results_list.append({
            "Batch Size": bs,
            "Final Val Accuracy": model.history['val_acc'][-1],
            "Training Time (s)": end_time - start_time
        })

    print("\n--- Batch Size Analysis Complete! ---")
    
    plots_dir = os.path.join(current_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    save_path = os.path.join(plots_dir, "C1_bs_analysis_curves.png")

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    for bs, history in bs_histories.items():
        plt.plot(history['val_loss'], label=f"BS = {bs}")
    plt.title("Validation Loss vs. Batch Size")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 0.5) 

    plt.subplot(1, 2, 2)
    for bs, history in bs_histories.items():
        plt.plot(history['val_acc'], label=f"BS = {bs}")
    plt.title("Validation Accuracy vs. Batch Size")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.ylim(0.8, 1.0) 

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Batch Size analysis plot saved to: {save_path}")

    print("\n--- Batch Size Performance Table ---")
    df_results = pd.DataFrame(results_list)
    print(df_results.to_markdown(index=False, floatfmt=".4f"))

if __name__ == "__main__":
    run_bs_analysis()