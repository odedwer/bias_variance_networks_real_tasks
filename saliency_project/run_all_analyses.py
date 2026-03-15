"""Run saliency analysis for a predefined set of model experiments.

This script launches `run_analysis.py` with the following argument pairs:

1. simpleCNN_bias1   -> models/SimpleCNN/bias=1.0/
2. simpleCNN_bias10  -> models/SimpleCNN/bias=10.0/
3. resnet_bias1      -> models/ResNet/bias=1.0/
4. resnet_bias10     -> models/ResNet/bias=10.0/

Usage:
    python run_all_analyses.py

If you want to run a subset of experiments, you can modify the `EXPERIMENTS` list below.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Path to the analysis script (relative to this script)
THIS_DIR = Path(__file__).resolve().parent
RUN_SCRIPT = THIS_DIR / "run_analysis.py"

# Experiments to run: (experiment_name, models_folder)
EXPERIMENTS = [
    ("simpleCNN_bias1", "models/SimpleCNN/bias=1.0/"),
    ("simpleCNN_bias10", "models/SimpleCNN/bias=10.0/"),
    ("resnet_bias1", "models/ResNet/bias=1.0/"),
    ("resnet_bias10", "models/ResNet/bias=10.0/"),
]


def run_experiment(experiment_name: str, models_folder: str) -> int:
    """Run a single analysis experiment and return the exit code."""
    cmd = [sys.executable, str(RUN_SCRIPT), "--experiment-name", experiment_name, "--models-folder", models_folder]
    print(f"\n=== Running: {experiment_name} (models_folder={models_folder}) ===")
    return subprocess.run(cmd, check=False).returncode


def main() -> None:
    for experiment_name, models_folder in EXPERIMENTS:
        ret = run_experiment(experiment_name, models_folder)
        if ret != 0:
            print(f"ERROR: Experiment '{experiment_name}' failed with exit code {ret}. Stopping.")
            sys.exit(ret)

    print("\nAll experiments completed successfully.")


if __name__ == "__main__":
    main()
