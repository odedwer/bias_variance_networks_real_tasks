import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = "runs"   # path to your tensorboard logs
TAG = "Accuracy/validation"  # change if your tag is different (maybe "Accuracy/val")

# ----------------------------
# Helper: read a single tfevent file
# ----------------------------
def load_scalar(event_file, tag=TAG):
    ea = EventAccumulator(event_file)
    ea.Reload()
    if tag not in ea.scalars.Keys():
        return None
    scalars = ea.scalars.Items(tag)
    steps = [s.step for s in scalars]
    values = [s.value for s in scalars]
    return np.array(values)

# ----------------------------
# Helper: detect plateau (very simple heuristic)
# plateau = very small improvement in last N epochs
# ----------------------------
def is_plateau(values, threshold=3):
    if len(values) < 2:
        return False
    return abs(values[-1] - values[0]) < threshold

# ----------------------------
# Helper: calculate training rate (accuracy change per epoch)
# ----------------------------
def calculate_training_rates(values, ranges):
    """Calculate avg training rate for given epoch ranges"""
    rates = {}
    for name, (start, end) in ranges.items():
        if end <= len(values):
            rate = (values[end-1] - values[start]) / (end - start)
            rates[name] = rate
    return rates

# ----------------------------
# Helper: compute final metrics per bias
# ----------------------------
def compute_final_metrics(arr):
    """Compute final accuracy stats and training rates"""
    final_acc = arr[:, -1]  # last epoch accuracy for each seed
    mean_final = final_acc.mean()
    std_final = final_acc.std()
    
    # Training rates over different epoch ranges
    ranges = {
        "first_20": (0, min(20, arr.shape[1])),
        "first_40": (0, min(40, arr.shape[1])),
        "last_20": (max(0, arr.shape[1] - 20), arr.shape[1]),
        "all": (0, arr.shape[1])
    }
    
    all_rates = {name: [] for name in ranges}
    for seed_data in arr:
        rates = calculate_training_rates(seed_data, ranges)
        for name, rate in rates.items():
            all_rates[name].append(rate)
    
    # Average rates across seeds
    avg_rates = {name: np.mean(rates) for name, rates in all_rates.items()}
    std_rates = {name: np.std(rates) for name, rates in all_rates.items()}
    
    return {
        "final_acc_mean": mean_final,
        "final_acc_std": std_final,
        "avg_rates": avg_rates,
        "std_rates": std_rates
    }

# ----------------------------
# Collect data
# ----------------------------
def collect_data(network_type_str = "ResNet"):
    bias_dict = {}

    for bias_folder in sorted(os.listdir(ROOT)):
        if not bias_folder.startswith("bias="):
            continue
        bias_path = os.path.join(ROOT, bias_folder)
        if not os.path.isdir(bias_path):
            continue
        
        all_runs = []
        excluded_runs = []
        print(f"Checking directory: {bias_path}")
        for root, dirs, files in os.walk(bias_path):
            if network_type_str in root:
                for f in files:
                    if "tfevents" in f:
                        data = load_scalar(os.path.join(root, f))
                        if data is not None:
                            if is_plateau(data):
                                excluded_runs.append(root)
                            else:
                                all_runs.append(data)
        
        # Print excluded runs
        if excluded_runs:
            print(f"  [EXCLUDED - PLATEAU] {network_type_str} runs:")
            for ex_run in excluded_runs:
                print(f"    {ex_run}")
        
        # Pad to same length (different training lengths)
        if all_runs:
            max_len = max(len(r) for r in all_runs)
            padded = [np.pad(r, (0, max_len - len(r)), 'edge') for r in all_runs]
            if any(len(r) < max_len for r in all_runs):
                print(f"  [PADDED] {network_type_str}: {len(all_runs)} seeds padded to length {max_len}")
            arr = np.vstack(padded)
            bias_dict[bias_folder] = arr
    
    return bias_dict

# ----------------------------
# Plot and display metrics
# ----------------------------
for network_type in ["ResNet", "SimpleCNN"]:
    bias_dict = collect_data(network_type_str=network_type)
    plt.figure(figsize=(12, 8))

    base_colors = ['red', 'blue', 'green', 'yellow']
    colors = [base_colors[i % len(base_colors)] for i in range(len(bias_dict))]

    print(f"\n{'='*70}")
    print(f"FINAL METRICS - {network_type}")
    print(f"{'='*70}")
    
    for (bias, arr), c in zip(bias_dict.items(), colors):
        mean = arr.mean(axis=0)
        min_v = arr.min(axis=0)
        max_v = arr.max(axis=0)

        epochs = np.arange(len(mean))

        # plot mean
        plt.plot(epochs, mean, label=bias, color=c, linewidth=1.5, zorder=3)
        # translucent fill
        plt.fill_between(epochs, min_v, max_v, color=c, alpha=0.15, zorder=1)
        # thin dashed lines for min and max on top of the fill so they remain visible
        plt.plot(epochs, min_v, color=c, linewidth=0.8, linestyle='--', alpha=0.9, zorder=4)
        plt.plot(epochs, max_v, color=c, linewidth=0.8, linestyle='--', alpha=0.9, zorder=4)
        
        # Compute and print metrics
        metrics = compute_final_metrics(arr)
        print(f"\n{bias}:")
        print(f"  Final Accuracy: {metrics['final_acc_mean']:.4f} ± {metrics['final_acc_std']:.4f}")
        print(f"  Training Rate (first 20 epochs): {metrics['avg_rates']['first_20']:.6f} ± {metrics['std_rates']['first_20']:.6f}")
        print(f"  Training Rate (first 40 epochs): {metrics['avg_rates']['first_40']:.6f} ± {metrics['std_rates']['first_40']:.6f}")
        print(f"  Training Rate (last 20 epochs): {metrics['avg_rates']['last_20']:.6f} ± {metrics['std_rates']['last_20']:.6f}")
        print(f"  Training Rate (all epochs): {metrics['avg_rates']['all']:.6f} ± {metrics['std_rates']['all']:.6f}")

    plt.title(f"Validation Accuracy in different biases (Averaged Across Seeds) - {network_type}")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"validation_accuracy_comparison_along_biases_{network_type}.png")
