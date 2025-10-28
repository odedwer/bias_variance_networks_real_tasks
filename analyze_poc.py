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

    epochs_for_compare = [
        (0, 0),  # Initial vs Initial
        (-1, -1),  # Final vs Final
        (0, -1),   # Initial vs Final
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 5),
        (0, 10),
        (1, -1),
        (2, -1),
        (3, -1),
        (5, -1),
        (10, -1)
    ]
    for epoch1, epoch2 in tqdm(epochs_for_compare):
        cka_comparison(
            epoch_idx1=epoch1,
            ma1=ma_low_var,
            ma_layers1=layer_names,
            epoch_idx2=epoch2,
            plot=True,
            show=False,
            save=True
        )
        cka_comparison(
            epoch_idx1=epoch1,
            ma1=ma_high_var,
            ma_layers1=layer_names,
            epoch_idx2=epoch2,
            plot=True,
            show=False,
            save=True
        )
        cka_comparison(
            epoch_idx1=epoch1,
            ma1=ma_low_var,
            ma_layers1=layer_names,
            epoch_idx2=epoch2,
            ma2=ma_high_var,
            ma_layers2=layer_names,
            plot=True,
            show=False,
            save=True
        )
        plt.close('all')



if __name__ == '__main__':
    run_analysis()