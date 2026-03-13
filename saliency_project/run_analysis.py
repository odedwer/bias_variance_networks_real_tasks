# run_analysis.py
"""
Main analysis script for computing and visualizing saliency metrics.

Processes trained models, computes saliency maps, extracts face landmarks,
and calculates metrics for saliency distribution analysis.
"""

import torch
import pandas as pd
from pathlib import Path
import sys
from pathlib import Path
import os

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from compute_saliency import compute_saliency
from face_parts.landmarks import FaceLandmarkDetector
from face_parts.masks import build_face_masks, save_masks, load_masks
from metrics import *
from plots import visualize_saliency_row
from face_recognition_model_comparison import FER2013Dataset, SimpleCNN, test_transforms
from ResNet import ResNet
from utils import get_device
from PIL import Image

SAL_DIR = Path("saliency_maps/resnet_bias10")
MASK_DIR = Path("saliency_project/face_parts/face_masks")
# Create output directory for visualizations
VIZ_DIR = Path("saliency_visualizations/resnet_bias10")
VIZ_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = "saliency_metrics_resnet_bias10.csv"
MODELS_FOLDER_PATH = "models/ResNet/bias=10.0/"
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
for model_chk_path in sorted(os.listdir(MODELS_FOLDER_PATH), key=lambda x: next((part.strip() for part in x.split(',') if 'seed' in part.lower()), x)):
    model_name, model = create_model(model_chk_path, device)
    models[model_name] = model

detector = FaceLandmarkDetector()
records = []
# --- PRECOMPUTE MASKS ---

for idx in range(len(dataset)):

    #convert tensor to PIL image for media pipe
    image_path = dataset.images[idx]
    image_id = Path(image_path).stem
    raw_img = Image.open(image_path).convert("RGB")
    raw_img = np.array(raw_img)

    mask_path = MASK_DIR / f"{idx}.pt"
    if mask_path.exists():
        continue

    landmarks = detector.detect(raw_img)
    if landmarks is None:
        continue

    masks = build_face_masks(raw_img, landmarks)
    save_masks(masks, mask_path)

# --- COMPUTE SALIENCY ---
for model_name, model in models.items():
    print(f"Processing model: {model_name}")
    for image_id, (image, label) in enumerate(dataset):
        path = SAL_DIR / model_name / f"{image_id}.pt"
        image = image.to(device)
        if path.exists():
            continue

        S = compute_saliency(model, image, label)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(S, path)

# --- GLOBAL THRESHOLD ---
# all_vals = torch.cat([
#     torch.load(p).flatten()
#     for p in SAL_DIR.rglob("*.pt")
# ])
# threshold = all_vals.quantile(0.95).item()
# print(f"Global saliency threshold (95th percentile): {threshold:.4f}")

# --- METRICS ---
for model_dir in SAL_DIR.iterdir():
    for sal_path in model_dir.glob("*.pt"):
        image_id = sal_path.stem
        
        mask_path = MASK_DIR / f"{image_id}.pt"
        if not mask_path.exists():
            continue
        #all goes to cpu for metric computation
        S = torch.load(sal_path, map_location="cpu")
        masks = load_masks(mask_path)
        S = S.cpu()
        masks = {k: v.cpu() for k, v in masks.items()}

        rec = {
            "model": model_dir.name,
            "image": image_id,
            # "normalized_entropy": saliency_entropy(S),
            # "max_short_distance": maxmean_short_distance(S, threshold)[0],
            # "mean_short_distance": maxmean_short_distance(S, threshold)[1],
        }

        THRESHOLDS = [0.2, 0.3, 0.4 ,0.5, 0.6]
        for threshold in THRESHOLDS:
            
            # Cluster-based metrics (all three methods)
            rec.update(connected_component_analysis(S, threshold))
        
        rec.update(face_part_coverage(S, masks, 0.3))
        rec.update(saliency_attribution(S, masks))
        records.append(rec)

df = pd.DataFrame(records)

df.to_csv(OUTPUT_FILE, index=False)

# --- VISUALIZATION ---
example_images = df["image"].unique()[:50]

for image_id in example_images:
    print(f"Visualizing image {image_id}")
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

    image = dataset[int(image_id)][0].permute(1, 2, 0)
    masks = load_masks(MASK_DIR / f"{image_id}.pt")

    save_path = VIZ_DIR / f"saliency_{image_id}.png"
    visualize_saliency_row(image, saliency_maps, masks, metrics, model_names, save_path=save_path)

print(f"\nAll visualizations saved to {VIZ_DIR}/")