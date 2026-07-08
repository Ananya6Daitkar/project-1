# evaluate.py
#
# runs the trained models on the test set and reports:
#   - precision, recall, F1 per class
#   - crater size distribution
#   - what kinds of things the model misses (failure analysis)
#   - inference speed
#
# Usage:
#   python src/evaluate.py --model yolov8n
#   python src/evaluate.py --all

import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    PROJECT_ROOT, PROCESSED_DIR, MODELS_DIR, ASSETS_DIR,
    CLASS_NAMES, CLASS_COLORS, yolo_to_xyxy, compute_iou, log
)

DATASET_YAML   = PROJECT_ROOT / "data" / "dataset.yaml"
IOU_THRESHOLD  = 0.5
CONF_THRESHOLD = 0.25


# ----------------------------------------------------------------
# load ground truth labels
# ----------------------------------------------------------------

def load_ground_truth(lbl_dir):
    """reads all .txt label files, returns {filename: [(cls, cx, cy, w, h), ...]}"""
    gt = {}
    for lp in sorted(lbl_dir.glob("*.txt")):
        boxes = []
        with open(lp) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    boxes.append((int(parts[0]), *map(float, parts[1:])))
        gt[lp.stem] = boxes
    return gt


def run_inference(model_name, img_dir, conf=CONF_THRESHOLD):
    """runs the model on every image in img_dir, times each one"""
    try:
        from ultralytics import YOLO
    except ImportError:
        log.error("ultralytics not installed")
        sys.exit(1)

    weights = MODELS_DIR / f"{model_name}_crater" / "weights" / "best.pt"
    if not weights.exists():
        log.error(f"weights not found at {weights} — train first")
        sys.exit(1)

    model  = YOLO(str(weights))
    images = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg"))
    predictions, timings = {}, []

    log.info(f"running {model_name} on {len(images)} test images...")
    for ip in tqdm(images, desc="inference"):
        img = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        H, W = img.shape

        t0      = time.perf_counter()
        results = model(ip, conf=conf, verbose=False)
        timings.append((time.perf_counter() - t0) * 1000)

        boxes = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls  = int(box.cls[0])
                conf_ = float(box.conf[0])
                # store as normalised coords
                cx = ((x1 + x2) / 2) / W
                cy = ((y1 + y2) / 2) / H
                w  = (x2 - x1) / W
                h  = (y2 - y1) / H
                boxes.append((cls, cx, cy, w, h, conf_))
        predictions[ip.stem] = boxes

    avg_ms = float(np.mean(timings)) if timings else 0.0
    log.info(f"  avg inference: {avg_ms:.1f} ms/image")
    return predictions, avg_ms


# ----------------------------------------------------------------
# metrics
# ----------------------------------------------------------------

def match_detections(gt_boxes, pred_boxes, img_w=256, img_h=256):
    """matches predictions to ground truth boxes using IoU, returns TP/FP/FN"""
    if not gt_boxes:
        return 0, len(pred_boxes), 0
    if not pred_boxes:
        return 0, 0, len(gt_boxes)

    matched_gt = set()
    tp = fp = 0

    # check highest-confidence predictions first
    sorted_preds = sorted(pred_boxes, key=lambda x: x[5] if len(x) > 5 else 1.0, reverse=True)

    for pred in sorted_preds:
        cls_p, cx_p, cy_p, w_p, h_p = pred[:5]
        px1, py1, px2, py2 = yolo_to_xyxy(cx_p, cy_p, w_p, h_p, img_w, img_h)

        best_iou, best_idx = 0.0, -1
        for i, gt in enumerate(gt_boxes):
            if i in matched_gt:
                continue
            cls_g, cx_g, cy_g, w_g, h_g = gt[:5]
            gx1, gy1, gx2, gy2 = yolo_to_xyxy(cx_g, cy_g, w_g, h_g, img_w, img_h)
            iou = compute_iou((px1, py1, px2, py2), (gx1, gy1, gx2, gy2))
            if iou > best_iou:
                best_iou, best_idx = iou, i

        if best_iou >= IOU_THRESHOLD and best_idx >= 0:
            tp += 1
            matched_gt.add(best_idx)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


def compute_metrics(ground_truth, predictions):
    """computes P/R/F1 per class and overall across the whole test set"""
    class_stats = {c: {"tp": 0, "fp": 0, "fn": 0} for c in range(3)}
    overall_tp = overall_fp = overall_fn = 0

    for stem, gt_boxes in ground_truth.items():
        pred_boxes = predictions.get(stem, [])

        for cls in range(3):
            gt_c = [b for b in gt_boxes  if b[0] == cls]
            pr_c = [b for b in pred_boxes if b[0] == cls]
            tp, fp, fn = match_detections(gt_c, pr_c)
            class_stats[cls]["tp"] += tp
            class_stats[cls]["fp"] += fp
            class_stats[cls]["fn"] += fn

        tp, fp, fn = match_detections(gt_boxes, pred_boxes)
        overall_tp += tp
        overall_fp += fp
        overall_fn += fn

    results = {}
    for cls, s in class_stats.items():
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        results[CLASS_NAMES[cls]] = {
            "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn,
        }

    p = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0.0
    r = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0.0
    results["overall"] = {
        "precision": round(p, 4),
        "recall":    round(r, 4),
        "f1":        round(2*p*r/(p+r) if p+r > 0 else 0, 4),
    }
    return results


def compute_size_distribution(predictions):
    """counts how many craters of each size class the model found"""
    counts   = {name: 0 for name in CLASS_NAMES.values()}
    sizes_px = []

    for preds in predictions.values():
        for p in preds:
            counts[CLASS_NAMES[p[0]]] += 1
            sizes_px.append(p[3] * 256)  # width in pixels

    return {"counts": counts, "sizes_px": sizes_px}


def analyze_failures(ground_truth, predictions):
    """
    looks for the three most common failure modes:
      - small craters the model missed
      - duplicate boxes on the same crater (NMS failure)
      - low-confidence detections that are probably wrong
    """
    missed_small   = 0
    total_small_gt = 0
    overlap_count  = 0
    low_conf_count = 0

    for stem, gt_boxes in ground_truth.items():
        pred_boxes = predictions.get(stem, [])

        small_gt = [b for b in gt_boxes   if b[0] == 0]
        small_pr = [b for b in pred_boxes if b[0] == 0]
        total_small_gt += len(small_gt)
        _, _, fn = match_detections(small_gt, small_pr)
        missed_small += fn

        # count any two predictions that heavily overlap
        for i, p1 in enumerate(pred_boxes):
            for j, p2 in enumerate(pred_boxes):
                if i >= j:
                    continue
                b1 = yolo_to_xyxy(*p1[1:5], 256, 256)
                b2 = yolo_to_xyxy(*p2[1:5], 256, 256)
                if compute_iou(b1, b2) > 0.5:
                    overlap_count += 1

        low_conf_count += sum(1 for p in pred_boxes if len(p) > 5 and p[5] < 0.35)

    return {
        "missed_small_craters":      missed_small,
        "total_small_gt":            total_small_gt,
        "small_recall":              round(1 - missed_small / total_small_gt, 3) if total_small_gt > 0 else None,
        "overlapping_detections":    overlap_count,
        "low_confidence_detections": low_conf_count,
    }


# ----------------------------------------------------------------
# report figure
# ----------------------------------------------------------------

def plot_evaluation_report(metrics, size_dist, failures, model_name, save_dir):
    """saves a 2x2 evaluation summary figure"""
    save_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(14, 10), facecolor="#0d1117")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # top-left: precision & recall bars per class
    ax1     = fig.add_subplot(gs[0, 0])
    classes = [k for k in metrics if k != "overall"]
    p_vals  = [metrics[c]["precision"] for c in classes]
    r_vals  = [metrics[c]["recall"]    for c in classes]
    x       = np.arange(len(classes))
    b1 = ax1.bar(x - 0.2, p_vals, 0.38, label="Precision", color="#3498db")
    b2 = ax1.bar(x + 0.2, r_vals, 0.38, label="Recall",    color="#e74c3c")
    ax1.set_xticks(x)
    ax1.set_xticklabels([c.replace("_crater", "") for c in classes], color="white")
    ax1.set_ylim(0, 1.15)
    ax1.set_title("Precision & Recall per class", color="white", fontsize=11)
    ax1.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
    ax1.set_facecolor("#1a1a2e")
    ax1.tick_params(colors="white")
    ax1.bar_label(b1, fmt="%.2f", padding=2, fontsize=8, color="white")
    ax1.bar_label(b2, fmt="%.2f", padding=2, fontsize=8, color="white")
    for sp in ax1.spines.values():
        sp.set_edgecolor("#444")

    # top-right: histogram of detected crater sizes
    ax2   = fig.add_subplot(gs[0, 1])
    sizes = size_dist["sizes_px"]
    if sizes:
        ax2.hist(sizes, bins=30, color="#2ecc71", edgecolor="#1a1a2e", alpha=0.85)
    ax2.set_title("Detected crater diameters (px)", color="white", fontsize=11)
    ax2.set_xlabel("diameter (px @ 256px tile)", color="white")
    ax2.set_ylabel("count", color="white")
    ax2.set_facecolor("#1a1a2e")
    ax2.tick_params(colors="white")
    for sp in ax2.spines.values():
        sp.set_edgecolor("#444")

    # bottom-left: class count bars
    ax3    = fig.add_subplot(gs[1, 0])
    counts = size_dist["counts"]
    bars   = ax3.bar(counts.keys(), counts.values(),
                     color=["#e74c3c", "#2ecc71", "#3498db"], edgecolor="#0d1117")
    ax3.bar_label(bars, padding=3, color="white", fontsize=9)
    ax3.set_title("detections by class", color="white", fontsize=11)
    ax3.set_facecolor("#1a1a2e")
    ax3.tick_params(colors="white", axis="x", labelrotation=15)
    for sp in ax3.spines.values():
        sp.set_edgecolor("#444")

    # bottom-right: text summary of failures
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    ax4.set_facecolor("#1a1a2e")
    lines = [
        f"model: {model_name.upper()}",
        "",
        f"  precision : {metrics['overall']['precision']:.3f}",
        f"  recall    : {metrics['overall']['recall']:.3f}",
        f"  f1        : {metrics['overall']['f1']:.3f}",
        "",
        "failure analysis",
        f"  missed small craters : {failures['missed_small_craters']} / {failures['total_small_gt']}",
        f"  small recall         : {failures.get('small_recall', 'n/a')}",
        f"  overlapping boxes    : {failures['overlapping_detections']}",
        f"  low-conf dets (<.35) : {failures['low_confidence_detections']}",
        "",
        "known weak spots",
        "  • clustered / overlapping craters",
        "  • low-contrast rims",
        "  • tiny craters under 5px",
    ]
    ax4.text(0.04, 0.96, "\n".join(lines),
             transform=ax4.transAxes, color="white", fontsize=9.5,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(facecolor="#1a1a2e", edgecolor="#444", boxstyle="round"))

    fig.suptitle(f"evaluation — {model_name.upper()}", color="white", fontsize=14, fontweight="bold")

    out_path = save_dir / f"eval_report_{model_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    log.info(f"report saved → {out_path}")
    return out_path


# ----------------------------------------------------------------
# main
# ----------------------------------------------------------------

def evaluate_model(model_name):
    test_img_dir = PROCESSED_DIR / "images" / "test"
    test_lbl_dir = PROCESSED_DIR / "labels" / "test"

    if not test_img_dir.exists():
        log.error("test set not found — run data_pipeline.py first")
        sys.exit(1)

    log.info(f"\nevaluating {model_name.upper()}...")
    ground_truth           = load_ground_truth(test_lbl_dir)
    predictions, avg_ms    = run_inference(model_name, test_img_dir)
    gt_filtered            = {k: v for k, v in ground_truth.items() if k in predictions}
    metrics                = compute_metrics(gt_filtered, predictions)
    size_dist              = compute_size_distribution(predictions)
    failures               = analyze_failures(gt_filtered, predictions)

    for cls, m in metrics.items():
        log.info(f"  {cls:<18}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")
    log.info(f"  avg inference: {avg_ms:.1f} ms")
    log.info(f"  missed small craters: {failures['missed_small_craters']}")
    log.info(f"  overlapping boxes:    {failures['overlapping_detections']}")

    out_dir = MODELS_DIR / f"{model_name}_crater"
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_result = {
        "model": model_name, "metrics": metrics,
        "size_distribution": size_dist["counts"],
        "failures": failures, "avg_inference_ms": round(avg_ms, 2),
    }
    with open(out_dir / "eval_results.json", "w") as f:
        json.dump(eval_result, f, indent=2)

    plot_evaluation_report(metrics, size_dist, failures, model_name, ASSETS_DIR)
    return eval_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluate crater models")
    parser.add_argument("--model", type=str, default="yolov8n",
                        choices=["yolov8n", "yolov8s"])
    parser.add_argument("--all",   action="store_true")
    args = parser.parse_args()

    if args.all:
        results = [evaluate_model(m) for m in ["yolov8n", "yolov8s"]]
        log.info("\n--- final comparison ---")
        log.info(f"{'metric':<22} {'yolov8n':>12} {'yolov8s':>12}")
        for key in ["precision", "recall", "f1"]:
            vals = [r["metrics"]["overall"][key] for r in results]
            log.info(f"  {key:<20} {vals[0]:>12.3f} {vals[1]:>12.3f}")
        log.info(f"  {'inference_ms':<20} "
                 f"{results[0]['avg_inference_ms']:>12.1f} "
                 f"{results[1]['avg_inference_ms']:>12.1f}")
    else:
        evaluate_model(args.model)
