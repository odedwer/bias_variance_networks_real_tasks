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
def is_plateau(values, threshold=0.03):
    if len(values) < 2:
        return False
    return abs(values[-1] - values[0]) < threshold

# ----------------------------
# Collect data
# ----------------------------
bias_dict = {}

for bias_folder in sorted(os.listdir(ROOT)):
    if not bias_folder.startswith("bias_"):
        continue
    bias_path = os.path.join(ROOT, bias_folder)
    if not os.path.isdir(bias_path):
        continue
    
    all_runs = []
    for root, dirs, files in os.walk(bias_path):
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

# ----------------------------
# Plot
# ----------------------------
plt.figure(figsize=(12, 8))

for bias, arr in bias_dict.items():
    mean = arr.mean(axis=0)
    min_v = arr.min(axis=0)
    max_v = arr.max(axis=0)

    epochs = np.arange(len(mean))

    plt.plot(epochs, mean, label=bias)          # average curve
    plt.fill_between(epochs, min_v, max_v, alpha=0.2)  # min-max shaded area

plt.title("Validation Accuracy vs Epochs (Averaged Across Seeds)")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("validation_accuracy_comparison_along_biases.png")
