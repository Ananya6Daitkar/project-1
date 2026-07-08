# utils.py — stuff used everywhere else

import os
import random
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import cv2
import matplotlib.pyplot as plt

# clean single-line logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("crater")

# --- folder paths ---
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
DATA_DIR      = PROJECT_ROOT / "data"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR    = PROJECT_ROOT / "models"
ASSETS_DIR    = PROJECT_ROOT / "app" / "assets"

for d in [RAW_DIR, PROCESSED_DIR, MODELS_DIR, ASSETS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- class setup ---
CLASS_NAMES = {0: "small_crater", 1: "medium_crater", 2: "large_crater"}
CLASS_COLORS = {
    0: (255, 80,  80),   # red   — small
    1: (80,  200, 80),   # green — medium
    2: (80,  140, 255),  # blue  — large
}
CLASS_THRESHOLDS = {
    "small":  (0.0,  0.10),
    "medium": (0.10, 0.25),
    "large":  (0.25, 1.00),
}


def seed_everything(seed: int = 42):
    """fix all random seeds so runs are reproducible"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    """YOLO normalised box → pixel corners"""
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)


def xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h):
    """pixel corners → YOLO normalised box"""
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h
    return cx, cy, w, h


def classify_crater_size(w_norm: float, h_norm: float) -> int:
    """returns 0/1/2 for small/medium/large based on box size"""
    size = max(w_norm, h_norm)
    if size < CLASS_THRESHOLDS["medium"][0]:
        return 0
    elif size < CLASS_THRESHOLDS["large"][0]:
        return 1
    return 2


def draw_detections(
    image: np.ndarray,
    boxes: List[Tuple],
    labels: Optional[List[int]] = None,
    confidences: Optional[List[float]] = None,
    thickness: int = 2,
) -> np.ndarray:
    """draws boxes + labels on a copy of the image, returns it"""
    img = image.copy()
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cls   = labels[i] if labels else 1
        conf  = confidences[i] if confidences else None
        color = CLASS_COLORS.get(cls, (200, 200, 200))

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        text = CLASS_NAMES.get(cls, "crater")
        if conf is not None:
            text += f" {conf:.2f}"

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def compute_iou(box1: Tuple, box2: Tuple) -> float:
    """standard IoU between two (x1,y1,x2,y2) boxes"""
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    a1    = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2    = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def plot_class_distribution(counts: Dict[str, int], save_path: Optional[Path] = None):
    """quick bar chart of how many craters per class"""
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.keys(), counts.values(),
                  color=["#e74c3c", "#2ecc71", "#3498db"], edgecolor="white")
    ax.bar_label(bars, padding=4, fontsize=11)
    ax.set_title("Crater Size Distribution", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_xlabel("Size Class")
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#16213e")
    ax.title.set_color("white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
