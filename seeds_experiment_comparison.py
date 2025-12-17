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
    # print(f"Loaded {len(values)} values from {event_file}")
    # print(f"Values: {values}")
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
# Collect data
# ----------------------------
def collect_data(network_type_str = "ResNet"):
    bias_dict = {}

    for bias_folder in sorted(os.listdir(ROOT)):
        #print(f"Processing bias folder: {bias_folder}")
        if not bias_folder.startswith("bias="):
            continue
        bias_path = os.path.join(ROOT, bias_folder)
        if not os.path.isdir(bias_path):
            continue
        
        all_runs = []
        print(f"Checking directory: {bias_path}")
        for root, dirs, files in os.walk(bias_path):
            if network_type_str in root:
                #print(f"  Found network directory: {root}")
                for f in files:
                    if "tfevents" in f:
                        data = load_scalar(os.path.join(root, f))
                        if data is not None and not is_plateau(data):
                            all_runs.append(data)
        
        # Pad to same length (different training lengths)
        if all_runs:
            max_len = max(len(r) for r in all_runs)
            padded = [np.pad(r, (0, max_len - len(r)), 'edge') for r in all_runs]
            arr = np.vstack(padded)
            bias_dict[bias_folder] = arr
    
    return bias_dict

# ----------------------------
# Plot
# ----------------------------
for network_type in ["ResNet", "SimpleCNN"]:
    bias_dict = collect_data(network_type_str=network_type)
    plt.figure(figsize=(12, 8))

 # choose a colormap and sample color for each bias to keep colors distinct and consistent
    # use simple default palette and repeat if there are more biases than colors
    base_colors = ['red', 'blue', 'green', 'yellow']
    colors = [base_colors[i % len(base_colors)] for i in range(len(bias_dict))]

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

    plt.title(f"Validation Accuracy in different biases (Averaged Across Seeds) - {network_type}")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"validation_accuracy_comparison_along_biases_{network_type}.png")
