import os
import sys
import torch
import torch.nn as nn 
import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
mlp_src_path = os.path.join(project_root, 'MLP', 'src')
lc_src_path = os.path.join(project_root, 'linear_classifiers', 'src')
sys.path.append(mlp_src_path)
sys.path.append(lc_src_path)

from utils import load_transform_split_mnist, per_class_accuracy
from neural_network_scratch import NeuralNetworkScratch
from logistic_regression_scratch import LogisticRegressionScratch
from softmax_regression_scratch import SoftmaxRegressionScratch

def save_confusion_matrix(y_true, y_pred, save_path, classes=[str(i) for i in range(10)]):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")
    plt.savefig(save_path)
    plt.close() 
    return cm

def save_misclassified(y_test, y_pred_test, test_set_obj, save_path):
    y_test = np.array(y_test)
    y_pred_test = np.array(y_pred_test)
    misclassified_indices = np.where(y_pred_test != y_test)[0]
    
    plt.figure(figsize=(10, 5))
    for i, img_idx in enumerate(misclassified_indices[:10]): 
        image, true_label = test_set_obj[img_idx]
        pred_label = y_pred_test[img_idx]
        
        plt.subplot(2, 5, i + 1)
        plt.imshow(image.squeeze(), cmap='gray')
        plt.title(f"True: {true_label}\nPred: {pred_label}")
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def evaluate_softmax_test(model, data_loader):
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in data_loader:
            X = X.float().view(X.shape[0], -1)
            preds = model.predict(X)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total


def run_final_analysis():
    print("--- C2: Starting Final Model Comparison ---")
    
    BEST_LR = 0.01
    BEST_BS = 64
    BEST_H1 = 128
    BEST_H2 = 64
    BEST_EPOCHS = 20
    
    print("Loading datasets...")
    train_loader, val_loader, test_loader, _, _, test_set_obj, _, _, _ = load_transform_split_mnist(
        classes=None, batch_size=BEST_BS
    )
    
    lr_train_loader, _, lr_test_loader, _, _, _, _, _, _ = load_transform_split_mnist(
        classes=[0, 1], batch_size=BEST_BS
    )
    
    results_list = []
    
    print("\nTraining Logistic Regression (on 0s and 1s)...")
    model_lr = LogisticRegressionScratch(input_dim=784, lr=0.01)
    start_time = time.time()
    model_lr.fit(lr_train_loader, epochs=10) 
    end_time = time.time()
    
    _, lr_test_acc = model_lr.evaluate(lr_test_loader)
    results_list.append({
        "Model": "Logistic Regression (0 vs 1)",
        "Test Accuracy": lr_test_acc,
        "Training Time (s)": end_time - start_time,
        "Complexity": "Very Low (~7.8K params)"
    })
    
    print("\nTraining Softmax Regression (on 0-9)...")
    model_softmax = SoftmaxRegressionScratch(input_dim=784, num_classes=10, lr=0.01)
    start_time = time.time()
    model_softmax.fit(train_loader, epochs=BEST_EPOCHS)
    end_time = time.time()
    
    softmax_test_acc = evaluate_softmax_test(model_softmax, test_loader)
    results_list.append({
        "Model": "Softmax Regression (0-9)",
        "Test Accuracy": softmax_test_acc,
        "Training Time (s)": end_time - start_time,
        "Complexity": "Low (~7.9K params)"
    })
    
    print("\nTraining Best Neural Network (on 0-9)...")
    best_nn_model = NeuralNetworkScratch(784, BEST_H1, BEST_H2, 10)
    start_time = time.time()
    best_nn_model.fit(train_loader, val_loader, epochs=BEST_EPOCHS, lr=BEST_LR)
    end_time = time.time()
    
    y_test, y_pred_test = [], []
    best_nn_model.eval()
    with torch.no_grad():
        for X, y in test_loader:
            preds = best_nn_model.predict(X)
            y_test.extend(y.numpy())
            y_pred_test.extend(preds.cpu().numpy())
            
    nn_test_acc = accuracy_score(y_test, y_pred_test)
    nn_params = sum(p.numel() for p in best_nn_model.parameters() if p.requires_grad)
    
    results_list.append({
        "Model": "Best Neural Network (0-9)",
        "Test Accuracy": nn_test_acc,
        "Training Time (s)": end_time - start_time,
        "Complexity": f"High (~{nn_params/1000:.0f}K params)"
    })
    
    print("\n" + "="*50)
    print("C2: Comprehensive Performance Summary Table")
    print("="*50)
    df_results = pd.DataFrame(results_list)
    print(df_results.to_markdown(index=False, floatfmt=".4f"))
    
    print("\n--- C2: Analysis ---")
    print("Logistic:   Extremely fast, but only for simple binary problems.")
    print("Softmax:    Fast baseline for multiclass, but is linear and has lower accuracy.")
    print("Neural Net: Slower to train (more params), but much more powerful and accurate for complex data.")

    print("\n" + "="*50)
    print("C2: Best Model Evaluation on Test Set")
    print("="*50)
    plots_dir = os.path.join(current_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    cm_path = os.path.join(plots_dir, "C2_final_confusion_matrix.png")
    print(f"Saving Confusion Matrix to {cm_path}...")
    cm = save_confusion_matrix(y_test, y_pred_test, cm_path)

    acc_path = os.path.join(plots_dir, "C2_final_per_class_acc.png")
    print(f"\nGenerating Per-Class Accuracy for Test Set...")

    df_acc = per_class_accuracy(cm)
    print(f"Saving Per-Class Accuracy to {acc_path}...")
    plt.figure(figsize=(8, 4))
    plt.bar(df_acc["Class"], df_acc["Accuracy"])
    plt.xlabel("Digit Class")
    plt.ylabel("Accuracy")
    plt.title("Final NN Model - Per-Class Accuracy")
    plt.savefig(acc_path)
    plt.close()

    misc_path = os.path.join(plots_dir, "C2_final_misclassified.png")
    print(f"Saving Misclassified Examples to {misc_path}...")
    save_misclassified(y_test, y_pred_test, test_set_obj, misc_path)
    
    print("\n--- Final Analysis Complete. Check the 'part_c/plots' folder. ---")

if __name__ == "__main__":
    run_final_analysis()