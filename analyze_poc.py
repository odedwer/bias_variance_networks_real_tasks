import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
# Assuming your original files are in the same directory
from model_analysis import ModelAnalysis, cka_comparison
from utils import get_device


def run_analysis():
    # !!! IMPORTANT !!!
    # UPDATE bdthese paths with the timestamped folder names created by poc_bias_variance.py
    # Look inside your 'runs' or 'models' directory for folders starting with "POC_"
    LOW_VAR_EXP_DIR = "25-10-2025_07-48-57_POC_Low_Variance_Bias"
    HIGH_VAR_EXP_DIR = "25-10-2025_07-49-54_POC_High_Variance_Bias"

    device = get_device()

    # --- 1. Performance Comparison ---
    print("\n--- Performance Comparison ---")
    print("✅ The definitive test accuracy for each model was printed at the end of the `poc_bias_variance.py` script.")
    print("Please refer to that terminal output for the final performance numbers.")

    # --- 2. CKA Comparison ---
    print("\nRunning CKA comparison between the final models...")
    if not os.path.exists(os.path.join("models", LOW_VAR_EXP_DIR)):
        raise FileNotFoundError(f"Experiment directory not found: {LOW_VAR_EXP_DIR}. Please check the path.")
    if not os.path.exists(os.path.join("models", HIGH_VAR_EXP_DIR)):
        raise FileNotFoundError(f"Experiment directory not found: {HIGH_VAR_EXP_DIR}. Please check the path.")

    # We still need ModelAnalysis to load the models and dataloaders for CKA
    # Note: The 'happy'/'sad' test set is a subset of the full test set, so the dataloader is compatible.
    ma_low_var = ModelAnalysis(LOW_VAR_EXP_DIR, device)
    ma_high_var = ModelAnalysis(HIGH_VAR_EXP_DIR, device)

    layer_names = [name for name, module in ma_low_var.models[-1].named_modules() if
                   isinstance(module, (torch.nn.Conv2d, torch.nn.Linear))]
    cka_over_epochs_low = {
        'conv1-conv2':[],
        'conv2-fc1':[],
        'fc1-fc2':[]
    }
    cka_over_epochs_high = {
        'conv1-conv2':[],
        'conv2-fc1':[],
        'fc1-fc2':[]
    }
    for epoch in tqdm(len(ma_low_var.models)):
        res_low = cka_comparison(
            epoch_idx1=epoch,
            ma1=ma_low_var,
            ma_layers1=layer_names,
            epoch_idx2=epoch,
            plot=True,
            show=False,
            save=True
        )
        res_high = cka_comparison(
            epoch_idx1=epoch,
            ma1=ma_high_var,
            ma_layers1=layer_names,
            epoch_idx2=epoch,
            plot=True,
            show=False,
            save=True
        )

        # save cka results for specific layer pairs
        cka_over_epochs_low['conv1-conv2'].append(res_low.loc['conv1', 'conv2'])
        cka_over_epochs_low['conv2-fc1'].append(res_low.loc['conv2', 'fc1'])
        cka_over_epochs_low['fc1-fc2'].append(res_low.loc['fc1', 'fc2'])
        cka_over_epochs_high['conv1-conv2'].append(res_high.loc['conv1', 'conv2'])
        cka_over_epochs_high['conv2-fc1'].append(res_high.loc['conv2', 'fc1'])
        cka_over_epochs_high['fc1-fc2'].append(res_high.loc['fc1', 'fc2'])
        plt.close('all')
    # Plot CKA over epochs for both models
    epochs = list(range(1, len(ma_low_var.models) + 1))
    plt.figure(figsize=(10, 6))
    for layer_pair in cka_over_epochs_low.keys():
        plt.plot(epochs, cka_over_epochs_low[layer_pair], label=f'Low Var - {layer_pair}')
        plt.plot(epochs, cka_over_epochs_high[layer_pair], label=f'High Var - {layer_pair}', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('CKA Similarity')
    plt.title('CKA Similarity Over Epochs for Low and High Variance Models')
    plt.legend()
    os.makedirs("figures/cka_over_epochs", exist_ok=True)
    plt.savefig("figures/cka_over_epochs/cka_over_epochs_comparison.pdf")



if __name__ == '__main__':
    run_analysis()