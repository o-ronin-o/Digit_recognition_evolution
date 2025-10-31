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
from flexible_nn import FlexibleNN 

def run_arch_analysis():
    print("--- C1.3: Starting Architecture Analysis ---")
    
    INPUT_DIM = 784
    OUTPUT_DIM = 10
    EPOCHS = 20
    LEARNING_RATE = 0.01 
    BATCH_SIZE = 64

    results_list = []
    
    train_loader, val_loader, _, _, _, _ = load_transform_split_mnist(
        classes=None, 
        batch_size=BATCH_SIZE
    )
    
    print("\n--- Testing Number of Layers ---")
    layer_configs = {
        "2_layers (baseline)": [INPUT_DIM, 128, 64, OUTPUT_DIM],
        "3_layers": [INPUT_DIM, 128, 128, 64, OUTPUT_DIM],
        "4_layers": [INPUT_DIM, 128, 128, 64, 32, OUTPUT_DIM],
        "5_layers": [INPUT_DIM, 128, 64, 64, 32, 32, OUTPUT_DIM]
    }
    
    for name, config in layer_configs.items():
        print(f"\nTesting Architecture: {name}")
        model = FlexibleNN(config)
        
        start_time = time.time()
        model.fit(train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE)
        end_time = time.time()
        
        results_list.append({
            "Experiment": "Number of Layers",
            "Architecture": name,
            "Final Val Accuracy": model.history['val_acc'][-1],
            "Training Time (s)": end_time - start_time
        })

    print("\n--- Testing Neurons Per Layer (2 Hidden Layers) ---")
    neuron_configs = {
        "Small [64, 32]": (64, 32),
        "Baseline [128, 64]": (128, 64),
        "Medium [256, 128]": (256, 128),
        "Large [512, 256]": (512, 256)
    }

    for name, (h1, h2) in neuron_configs.items():
        print(f"\nTesting Architecture: {name}")
        model = NeuralNetworkScratch(INPUT_DIM, h1, h2, OUTPUT_DIM) 
        
        start_time = time.time()
        model.fit(train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE)
        end_time = time.time()
        
        results_list.append({
            "Experiment": "Neurons Per Layer",
            "Architecture": name,
            "Final Val Accuracy": model.history['val_acc'][-1],
            "Training Time (s)": end_time - start_time
        })

    print("\n--- Architecture Comparison Table ---")
    df_results = pd.DataFrame(results_list)
    print(df_results.to_markdown(index=False, floatfmt=".4f"))

if __name__ == "__main__":
    run_arch_analysis()