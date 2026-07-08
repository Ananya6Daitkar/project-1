# visualize.py
#
# everything you need to actually see what the model is doing:
#   - draw boxes + confidence scores on an image
#   - side-by-side before/after comparison
#   - confidence heatmap (shows where the model is paying attention)
#   - 4-panel full report figure
#
# Usage:
#   python src/visualize.py --model yolov8n --image path/to/img.png
#   python src/visualize.py --model yolov8n --demo   <- uses first 3 test images

import sys
import argparse
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    PROJECT_ROOT, PROCESSED_DIR, MODELS_DIR, ASSETS_DIR,
    CLASS_NAMES, CLASS_COLORS, yolo_to_xyxy, draw_detections, log
)

CONF_THRESHOLD = 0.25


def predict_image(model_name, image_path, conf=CONF_THRESHOLD):
    """runs YOLO on one image, returns (bgr_img, boxes, classes, confidences)"""
    try:
        from ultralytics import YOLO
    except ImportError:
        log.error("ultralytics not installed")
        sys.exit(1)

    weights = MODELS_DIR / f"{model_name}_crater" / "weights" / "best.pt"
    if not weights.exists():
        log.error(f"weights not found: {weights}")
        sys.exit(1)

    model  = YOLO(str(weights))
    img    = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"can't read {image_path}")

    H, W     = img.shape[:2]
    results  = model(image_path, conf=conf, verbose=False)
    boxes, classes, confidences = [], [], []

    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            boxes.append((x1, y1, x2, y2))
            classes.append(int(box.cls[0]))
            confidences.append(float(box.conf[0]))

    return img, boxes, classes, confidences


def make_before_after(image_path, model_name, save_path=None, conf=CONF_THRESHOLD):
    """puts raw image and annotated image side by side with a count banner"""
    img, boxes, classes, confs = predict_image(model_name, image_path, conf)

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    annotated     = draw_detections(img.copy(), boxes, classes, confs)
    class_counts  = {name: 0 for name in CLASS_NAMES.values()}
    for cls in classes:
        class_counts[CLASS_NAMES[cls]] += 1

    # build the canvas with a top banner
    banner_h = 40
    h, w     = img.shape[:2]
    canvas   = np.zeros((h + banner_h, w * 2 + 20, 3), dtype=np.uint8)
    canvas[:, :, :] = (13, 17, 23)

    canvas[banner_h:banner_h + h, :w]           = img
    canvas[banner_h:banner_h + h, w + 20:w*2+20] = annotated

    cv2.putText(canvas, "original", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    cv2.putText(canvas, f"detected  |  total: {len(boxes)} craters",
                (w + 30, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    for cls_id, name in CLASS_NAMES.items():
        color = CLASS_COLORS[cls_id]
        cv2.putText(canvas, f"{name.replace('_crater','')}: {class_counts[name]}",
                    (w + 30, banner_h + 18 + cls_id * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    if save_path:
        cv2.imwrite(str(save_path), canvas)
        log.info(f"before/after → {save_path}")
    return canvas


def make_confidence_heatmap(image_path, model_name, save_path=None, conf=CONF_THRESHOLD):
    """
    builds a heatmap by placing a Gaussian blob at each detection,
    scaled by the confidence score, then overlays it on the image.
    bright = model is very confident something's there.
    """
    img, boxes, classes, confs = predict_image(model_name, image_path, conf)
    H, W    = img.shape[:2]
    heatmap = np.zeros((H, W), dtype=np.float32)

    for (x1, y1, x2, y2), conf_val in zip(boxes, confs):
        cx     = (x1 + x2) // 2
        cy     = (y1 + y2) // 2
        radius = max((x2 - x1), (y2 - y1))
        k_size = max(radius * 2 + 1, 5)
        k_size = k_size if k_size % 2 == 1 else k_size + 1
        sigma  = radius / 2 or 5
        kernel    = cv2.getGaussianKernel(int(k_size), sigma)
        kernel_2d = kernel @ kernel.T * conf_val

        ky, kx  = kernel_2d.shape
        y0, x0  = cy - ky // 2, cx - kx // 2
        ys, ye  = max(0, y0), min(H, y0 + ky)
        xs, xe  = max(0, x0), min(W, x0 + kx)
        heatmap[ys:ye, xs:xe] += kernel_2d[ys-y0:ys-y0+(ye-ys), xs-x0:xs-x0+(xe-xs)]

    heat_u8    = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    overlay   = cv2.addWeighted(img, 0.55, heat_color, 0.45, 0)
    annotated = draw_detections(overlay, boxes, classes, confs)

    if save_path:
        cv2.imwrite(str(save_path), annotated)
        log.info(f"heatmap → {save_path}")
    return annotated


def visualize_full_report(image_path, model_name, save_path=None, conf=CONF_THRESHOLD):
    """2x2 figure: raw image | annotated | heatmap | size pie chart"""
    img, boxes, classes, confs = predict_image(model_name, image_path, conf)

    img_rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB if len(img.shape) == 3 else cv2.COLOR_GRAY2RGB)
    annotated  = draw_detections(img_rgb.copy(), boxes, classes, confs)
    heatmap    = make_confidence_heatmap(image_path, model_name, conf=conf)
    heat_rgb   = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    class_counts = {}
    for cls in classes:
        class_counts[CLASS_NAMES[cls]] = class_counts.get(CLASS_NAMES[cls], 0) + 1

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), facecolor="#0d1117")
    fig.suptitle(f"{model_name.upper()}  —  {len(boxes)} craters detected",
                 color="white", fontsize=13, fontweight="bold")

    for (data, title, row, col) in [
        (img_rgb,   "original",           0, 0),
        (annotated, "detections",         0, 1),
        (heat_rgb,  "confidence heatmap", 1, 0),
    ]:
        ax = axes[row][col]
        ax.imshow(data)
        ax.set_title(title, color="white", fontsize=11)
        ax.axis("off")
        ax.set_facecolor("#1a1a2e")

    # pie chart of class split
    ax_pie = axes[1][1]
    ax_pie.set_facecolor("#1a1a2e")
    if class_counts:
        pie_colors = [
            f"#{CLASS_COLORS[k][2]:02x}{CLASS_COLORS[k][1]:02x}{CLASS_COLORS[k][0]:02x}"
            for k in range(3) if CLASS_NAMES[k] in class_counts
        ]
        _, _, autotexts = ax_pie.pie(
            list(class_counts.values()),
            labels=list(class_counts.keys()),
            colors=pie_colors,
            autopct="%1.0f%%",
            textprops={"color": "white", "fontsize": 9},
        )
        for at in autotexts:
            at.set_color("white")
    else:
        ax_pie.text(0.5, 0.5, "no detections", ha="center", va="center",
                    color="white", fontsize=12)
    ax_pie.set_title("size breakdown", color="white", fontsize=11)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        log.info(f"report → {save_path}")
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="crater detection visualiser")
    parser.add_argument("--model", type=str, default="yolov8n")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--demo",  action="store_true")
    parser.add_argument("--conf",  type=float, default=CONF_THRESHOLD)
    args = parser.parse_args()

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    if args.demo or args.image is None:
        images = sorted((PROCESSED_DIR / "images" / "test").glob("*.png"))[:3]
        if not images:
            log.error("no test images — run data_pipeline.py first")
            sys.exit(1)
    else:
        images = [Path(args.image)]

    for ip in images:
        make_before_after(ip, args.model,
                          save_path=ASSETS_DIR / f"before_after_{ip.stem}.png",
                          conf=args.conf)
        visualize_full_report(ip, args.model,
                              save_path=ASSETS_DIR / f"report_{ip.stem}.png",
                              conf=args.conf)

    log.info("done — check app/assets/")
