"""
IRDDT - YOLOv8 Training Script for Google Colab
Infrared Drone Detection & Tracking

Copy-paste each section (marked with ═══) into separate Colab cells.
Run them one at a time, in order.
"""

# ═══════════════════════════════════════════════════════════════════
# CELL 1: Setup & Install
# ═══════════════════════════════════════════════════════════════════

# Install ultralytics (YOLOv8)
# !pip install ultralytics -q

# Verify GPU is available
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")


# ═══════════════════════════════════════════════════════════════════
# CELL 2: Mount Google Drive & Extract Dataset
# ═══════════════════════════════════════════════════════════════════

#from google.colab import drive
#drive.mount('/content/drive')

# Extract dataset from Drive to Colab runtime (faster I/O)
# NOTE: Update this path to match where you uploaded irddt_dataset.zip
import zipfile
import os

zip_path = "/content/drive/MyDrive/irddt_dataset.zip"  # <-- UPDATE THIS PATH if needed
extract_to = "/content/"

print(f"Extracting {zip_path}...")
with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall(extract_to)
print("Done!")

# Verify extraction
dataset_path = "/content/irddt_dataset"
for split in ["train", "val"]:
    imgs = len(os.listdir(os.path.join(dataset_path, "images", split)))
    lbls = len(os.listdir(os.path.join(dataset_path, "labels", split)))
    print(f"{split}: {imgs} images, {lbls} labels")


# ═══════════════════════════════════════════════════════════════════
# CELL 3: Visualize a Few Samples (Optional but Recommended)
# ═══════════════════════════════════════════════════════════════════

import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

def show_sample(img_path, lbl_path, ax):
    """Draw YOLO bbox on image."""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    with open(lbl_path) as f:
        content = f.read().strip()

    if content:
        parts = content.split()
        cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        # Convert back to pixel coords
        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, "drone", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        ax.set_title("DRONE PRESENT", color='green')
    else:
        ax.set_title("NO DRONE", color='red')

    ax.imshow(img)
    ax.axis('off')

# Show 8 random samples
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
img_dir = os.path.join(dataset_path, "images", "train")
lbl_dir = os.path.join(dataset_path, "labels", "train")
samples = random.sample(os.listdir(img_dir), 8)

for ax, fname in zip(axes.flat, samples):
    img_path = os.path.join(img_dir, fname)
    lbl_path = os.path.join(lbl_dir, fname.replace('.jpg', '.txt'))
    show_sample(img_path, lbl_path, ax)

plt.suptitle("IRDDT - Training Samples", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


# ═══════════════════════════════════════════════════════════════════
# CELL 4: Train YOLOv8n
# ═══════════════════════════════════════════════════════════════════

from ultralytics import YOLO

# Load pretrained YOLOv11 nano model (latest, fewer params, better accuracy)
model = YOLO("yolo11n.pt")

# Train on our thermal drone dataset
results = model.train(
    data="/content/irddt_dataset/dataset.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    patience=10,          # Early stopping: stop if no improvement for 10 epochs
    save=True,
    save_period=10,       # Save checkpoint every 10 epochs
    device=0,             # GPU
    workers=2,
    project="/content/irddt_runs",
    name="train_v1",
    # Augmentation tuned for thermal/IR images
    hsv_h=0.0,            # No hue shift (grayscale thermal images)
    hsv_s=0.0,            # No saturation shift
    hsv_v=0.2,            # Slight brightness variation
    degrees=10,           # Small rotation
    translate=0.1,
    scale=0.3,            # Scale variation (drones at different distances)
    fliplr=0.5,           # Horizontal flip
    flipud=0.0,           # No vertical flip (drones don't fly upside down)
    mosaic=0.5,           # Mosaic augmentation
    mixup=0.1,            # Light mixup
)

print("\nTraining complete!")
print("Best model: /content/irddt_runs/train_v1/weights/best.pt")


# ═══════════════════════════════════════════════════════════════════
# CELL 5: Evaluate the Model
# ═══════════════════════════════════════════════════════════════════

# Load the best model
best_model = YOLO("/content/irddt_runs/train_v1/weights/best.pt")

# Run validation
metrics = best_model.val(
    data="/content/irddt_dataset/dataset.yaml",
    imgsz=640,
    batch=16,
    device=0,
)

print("\n--- EVALUATION RESULTS ---")
print(f"mAP@50:      {metrics.box.map50:.4f}")
print(f"mAP@50-95:   {metrics.box.map:.4f}")
print(f"Precision:   {metrics.box.mp:.4f}")
print(f"Recall:      {metrics.box.mr:.4f}")


# ═══════════════════════════════════════════════════════════════════
# CELL 6: Test on a Few Images (Visual Check)
# ═══════════════════════════════════════════════════════════════════

# Run inference on random val images
val_img_dir = os.path.join(dataset_path, "images", "val")
test_images = random.sample(os.listdir(val_img_dir), 8)
test_paths = [os.path.join(val_img_dir, f) for f in test_images]

results = best_model.predict(
    source=test_paths,
    imgsz=640,
    conf=0.25,
    save=False,
)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for ax, result in zip(axes.flat, results):
    img = result.plot()  # Image with detections drawn
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    ax.axis('off')
    n_dets = len(result.boxes)
    ax.set_title(f"{n_dets} detection(s)", color='green' if n_dets > 0 else 'red')

plt.suptitle("IRDDT - YOLOv11n Predictions on Validation Set", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


# ═══════════════════════════════════════════════════════════════════
# CELL 7: Save Best Weights to Google Drive
# ═══════════════════════════════════════════════════════════════════

import shutil

# Create output directory on Drive
drive_save_dir = "/content/drive/MyDrive/IRDDT_models"
os.makedirs(drive_save_dir, exist_ok=True)

# Copy best weights
src = "/content/irddt_runs/train_v1/weights/best.pt"
dst = os.path.join(drive_save_dir, "yolo11n_irddt_best.pt")
shutil.copy2(src, dst)
print(f"Best model saved to: {dst}")

# Also copy last weights as backup
src_last = "/content/irddt_runs/train_v1/weights/last.pt"
dst_last = os.path.join(drive_save_dir, "yolo11n_irddt_last.pt")
shutil.copy2(src_last, dst_last)
print(f"Last model saved to: {dst_last}")

# Copy training results (curves, metrics)
results_dir = "/content/irddt_runs/train_v1"
for fname in os.listdir(results_dir):
    if fname.endswith(('.png', '.csv')):
        shutil.copy2(os.path.join(results_dir, fname), os.path.join(drive_save_dir, fname))
        print(f"Copied: {fname}")

print("\nAll done! You can now download yolov8n_irddt_best.pt for local inference.")
