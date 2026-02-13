# run_analysis.py

import torch
import pandas as pd
from pathlib import Path
import sys
from pathlib import Path
import os

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from compute_saliency import compute_saliency
# from face_parts.landmarks import FaceLandmarkDetector
# from face_parts.masks import build_face_masks, save_masks, load_masks
from metrics import *
from plots import visualize_saliency_row
from face_recognition_model_comparison import FER2013Dataset, SimpleCNN, test_transforms
from ResNet import ResNet
from utils import get_device

SAL_DIR = Path("saliency_maps")
MASK_DIR = Path("face_masks")
MODELS_FOLDER_PATH = "models/models_for_analysis_resnet109/"
classes=["fear","angry"]

dataset = FER2013Dataset('data/face-expression/test', transform=test_transforms, classes=classes)

def create_model(model_chk_path, device):
    model_chkpoints = os.listdir(os.path.join(MODELS_FOLDER_PATH, model_chk_path))
    model_chkpoints.sort(key=lambda x: os.path.getctime(os.path.join(MODELS_FOLDER_PATH, model_chk_path, x)))     # sort by creation date - all of the model's checkpoints
    if "SimpleCNN" in model_chk_path:
        model = SimpleCNN(bn="BN=True" in model_chk_path, init_bias=0.0 if "Bias=None" not in model_chk_path else None)
    elif "ResNet" in model_chk_path:
        model = ResNet(bn="BN=True" in model_chk_path, bias="Bias=None" not in model_chk_path)


    model.load_state_dict(torch.load(os.path.join(MODELS_FOLDER_PATH, model_chk_path, model_chkpoints[-1]), map_location=device))
    model = model.to(device)
    #set models to evaluation mode
    model.eval()

    return model_chk_path, model

device = get_device()
models = {}
for model_chk_path in os.listdir(MODELS_FOLDER_PATH):
    model_name, model = create_model(model_chk_path, device)
    models[model_name] = model

# detector = FaceLandmarkDetector()
records = []

# --- PRECOMPUTE MASKS ---
# for image_id, image in dataset:
#     mask_path = MASK_DIR / f"{image_id}.pt"
#     if mask_path.exists():
#         continue

#     landmarks = detector.detect(image)
#     if landmarks is None:
#         continue

#     masks = build_face_masks(image, landmarks)
#     save_masks(masks, mask_path)

# --- COMPUTE SALIENCY ---
for model_name, model in models.items():
    print(f"Processing model: {model_name}")
    for image_id, (image, label) in enumerate(dataset):
        path = SAL_DIR / model_name / f"{image_id}.pt"
        if path.exists():
            continue

        S = compute_saliency(model, image, label)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(S, path)

# --- GLOBAL THRESHOLD ---
all_vals = torch.cat([
    torch.load(p).flatten()
    for p in SAL_DIR.rglob("*.pt")
])
threshold = all_vals.quantile(0.9).item()

# --- METRICS ---
for model_dir in SAL_DIR.iterdir():
    for sal_path in model_dir.glob("*.pt"):
        image_id = sal_path.stem
        mask_path = MASK_DIR / f"{image_id}.pt"
        if not mask_path.exists():
            continue

        S = torch.load(sal_path)
        masks = load_masks(mask_path)

        rec = {
            "model": model_dir.name,
            "image": image_id,
            "entropy": saliency_entropy(S),
            "max_short_distance": max_short_distance(S, threshold),
        }
        rec.update(face_part_coverage(S, masks, threshold))
        records.append(rec)

df = pd.DataFrame(records)
df.to_csv("saliency_metrics.csv", index=False)

# --- VISUALIZATION ---
example_images = df["image"].unique()[:3]

for image_id in example_images:
    saliency_maps = []
    metrics = []
    model_names = []

    for model in list(models.keys())[:5]:
        S = torch.load(SAL_DIR / model / f"{image_id}.pt")
        saliency_maps.append(S)
        metrics.append(
            df[(df.image == image_id) & (df.model == model)].iloc[0].to_dict()
        )
        model_names.append(model)

    image = image_lookup[image_id]
    masks = load_masks(MASK_DIR / f"{image_id}.pt")

    visualize_saliency_row(image, saliency_maps, masks, metrics, model_names)