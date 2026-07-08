# train.py
#
# trains two YOLOv8 models and compares them:
#   yolov8n — nano, 3.2M params, fast (~8ms) — our baseline
#   yolov8s — small, 11.2M params, slower but more accurate
#
# Usage:
#   python src/train.py --model yolov8n --epochs 50
#   python src/train.py --model yolov8s --epochs 50
#   python src/train.py --all --epochs 50   <- trains both back to back

import argparse
import sys
import time
import json
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import PROJECT_ROOT, PROCESSED_DIR, MODELS_DIR, log, seed_everything

DATASET_YAML = PROJECT_ROOT / "data" / "dataset.yaml"

MODEL_CONFIGS = {
    "yolov8n": {
        "weights":     "yolov8n.pt",
        "out_dir":     MODELS_DIR / "yolov8n_crater",
        "description": "YOLOv8 Nano — 3.2M params, ~8ms, baseline",
        "img_size":    256,
        "batch":       16,
    },
    "yolov8s": {
        "weights":     "yolov8s.pt",
        "out_dir":     MODELS_DIR / "yolov8s_crater",
        "description": "YOLOv8 Small — 11.2M params, ~14ms, more accurate",
        "img_size":    256,
        "batch":       8,
    },
}

# small craters are underrepresented so we weight them higher in the loss
CLASS_WEIGHTS = [2.5, 1.0, 0.8]


def check_dataset_ready():
    """make sure the processed folders exist and aren't empty"""
    for split in ["train", "val"]:
        split_dir = PROCESSED_DIR / "images" / split
        if not split_dir.exists():
            return False
        if not list(split_dir.glob("*.png")) + list(split_dir.glob("*.jpg")):
            return False
    return True


def verify_dataset_yaml():
    """write a default dataset.yaml if it's missing"""
    if not DATASET_YAML.exists():
        log.warning("dataset.yaml not found — writing a default one")
        cfg = {
            "path":  str(PROCESSED_DIR),
            "train": "images/train",
            "val":   "images/val",
            "test":  "images/test",
            "nc":    3,
            "names": ["small_crater", "medium_crater", "large_crater"],
        }
        with open(DATASET_YAML, "w") as f:
            yaml.dump(cfg, f)
    return DATASET_YAML


def save_training_summary(model_name, results, elapsed):
    """save key metrics to JSON so we can compare later"""
    out_dir = MODEL_CONFIGS[model_name]["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        metrics = {
            "model":           model_name,
            "description":     MODEL_CONFIGS[model_name]["description"],
            "elapsed_minutes": round(elapsed / 60, 2),
            "map50":      float(results.box.map50) if hasattr(results, "box") else None,
            "map50_95":   float(results.box.map)   if hasattr(results, "box") else None,
            "precision":  float(results.box.mp)    if hasattr(results, "box") else None,
            "recall":     float(results.box.mr)    if hasattr(results, "box") else None,
        }
    except Exception:
        metrics = {"model": model_name, "elapsed_minutes": round(elapsed / 60, 2)}

    summary_path = out_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"summary saved → {summary_path}")
    return metrics


def train_model(model_name, epochs=50, resume=False):
    """trains one model variant, validates it, saves results"""
    try:
        from ultralytics import YOLO
    except ImportError:
        log.error("ultralytics not installed — pip install ultralytics")
        sys.exit(1)

    if model_name not in MODEL_CONFIGS:
        log.error(f"unknown model '{model_name}', pick from {list(MODEL_CONFIGS)}")
        sys.exit(1)

    cfg     = MODEL_CONFIGS[model_name]
    out_dir = cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"\ntraining {model_name.upper()} | epochs={epochs} batch={cfg['batch']} img={cfg['img_size']}")

    if not check_dataset_ready():
        log.error("dataset missing — run: python src/data_pipeline.py --robbins")
        sys.exit(1)

    yaml_path = verify_dataset_yaml()

    # resume from last checkpoint if asked
    last_weights = out_dir / "weights" / "last.pt"
    if resume and last_weights.exists():
        log.info(f"resuming from {last_weights}")
        model = YOLO(str(last_weights))
    else:
        model = YOLO(cfg["weights"])

    t0 = time.time()
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=cfg["img_size"],
        batch=cfg["batch"],
        project=str(MODELS_DIR),
        name=f"{model_name}_crater",
        exist_ok=True,
        patience=15,       # stop early if nothing's improving
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
        mosaic=0.9,        # augmentation — mix 4 tiles into one
        mixup=0.1,         # blend two images slightly
        degrees=10.0,      # small rotation
        flipud=0.3,
        fliplr=0.5,
        scale=0.5,
        translate=0.1,
        conf=0.25,         # low threshold helps catch small craters
        iou=0.45,
    )
    elapsed = time.time() - t0

    log.info("running final validation...")
    val_results = model.val(data=str(yaml_path), imgsz=cfg["img_size"])

    metrics = save_training_summary(model_name, val_results, elapsed)
    log.info(f"\ndone in {elapsed/60:.1f} min")
    if metrics.get("map50"):
        log.info(f"  mAP@50={metrics['map50']:.3f}  P={metrics.get('precision','?'):.3f}  R={metrics.get('recall','?'):.3f}")

    return metrics


def print_comparison_table():
    """prints a side-by-side table of all trained models"""
    rows = []
    for name, cfg in MODEL_CONFIGS.items():
        summary = cfg["out_dir"] / "training_summary.json"
        if summary.exists():
            with open(summary) as f:
                rows.append(json.load(f))

    if not rows:
        log.warning("no trained models found yet")
        return

    log.info("\n--- model comparison ---")
    log.info(f"{'model':<12} {'mAP@50':>8} {'precision':>10} {'recall':>8} {'time(min)':>10}")
    log.info("-" * 55)
    for r in rows:
        log.info(f"{r['model']:<12} "
                 f"{r.get('map50', 0):>8.3f} "
                 f"{r.get('precision', 0):>10.3f} "
                 f"{r.get('recall', 0):>8.3f} "
                 f"{r.get('elapsed_minutes', 0):>10.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train YOLOv8 crater detector")
    parser.add_argument("--model",   type=str, default="yolov8n",
                        choices=list(MODEL_CONFIGS))
    parser.add_argument("--all",     action="store_true", help="train both models")
    parser.add_argument("--epochs",  type=int, default=50)
    parser.add_argument("--resume",  action="store_true")
    parser.add_argument("--compare", action="store_true",
                        help="just print the comparison table")
    args = parser.parse_args()

    seed_everything(42)

    if args.compare:
        print_comparison_table()
    elif args.all:
        for name in MODEL_CONFIGS:
            train_model(name, epochs=args.epochs, resume=args.resume)
        print_comparison_table()
    else:
        train_model(args.model, epochs=args.epochs, resume=args.resume)
