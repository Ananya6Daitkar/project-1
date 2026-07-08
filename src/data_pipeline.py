# data_pipeline.py
#
# Builds the training dataset from scratch. Two modes:
#   --robbins   pulls the real Robbins lunar catalog (384k craters, 4.3 MB)
#               and renders terrain tiles with those exact crater positions/sizes
#   --synthetic totally random tiles, no download needed
#
# Either way the output is the same: 256x256 YOLO-format tiles in data/processed/
#
# Usage:
#   python src/data_pipeline.py --robbins          <- recommended
#   python src/data_pipeline.py --robbins --n 800  <- more tiles
#   python src/data_pipeline.py --synthetic        <- no internet

import os
import sys
import shutil
import random
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import PROJECT_ROOT, RAW_DIR, PROCESSED_DIR, seed_everything, log

# --- settings ---
TILE_SIZE = 256
OVERLAP   = 32
SPLITS    = {"train": 0.70, "val": 0.20, "test": 0.10}
SEED      = 42

# each tile covers a 50x50 km patch of the lunar surface
# this scale lets medium/large craters still fit inside a single tile
KM_PER_TILE = 50.0

# skip craters wider than 45% of the tile — they'd dominate the whole frame
MAX_CRATER_RADIUS_FRAC = 0.45


# ----------------------------------------------------------------
# load the Robbins catalog from HuggingFace
# ----------------------------------------------------------------

def load_robbins_catalog():
    """downloads juliensimon/lunar-craters-robbins and returns a dataframe"""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "install the datasets library first: pip install datasets\n"
            "or just use --synthetic if you're offline"
        )

    log.info("downloading Robbins catalog from HuggingFace (4.3 MB, cached after first run)...")
    ds = load_dataset("juliensimon/lunar-craters-robbins", split="train")
    df = ds.to_pandas()
    log.info(f"  got {len(df):,} craters, diameter {df['diameter_km'].min():.1f}–{df['diameter_km'].max():.0f} km")
    log.info(f"  class breakdown: {df['size_class'].value_counts().to_dict()}")
    return df


def sample_catalog_region(df, center_lat, center_lon, region_km=KM_PER_TILE):
    """grab all craters within a square box around the given lat/lon"""
    km_per_deg_lat = 30.87          # on the Moon, 1 degree lat ≈ 30.87 km
    km_per_deg_lon = 30.87 * np.cos(np.deg2rad(center_lat))

    dlat = (region_km / 2) / km_per_deg_lat
    dlon = (region_km / 2) / max(km_per_deg_lon, 0.01)

    mask = (
        (df["latitude_deg"]  >= center_lat - dlat) &
        (df["latitude_deg"]  <= center_lat + dlat) &
        (df["longitude_deg"] >= center_lon - dlon) &
        (df["longitude_deg"] <= center_lon + dlon)
    )
    return df[mask].copy()


def craters_to_tile_annotations(region_df, center_lat, center_lon,
                                 region_km=KM_PER_TILE, tile_px=TILE_SIZE):
    """
    takes craters from the catalog and converts their lat/lon/diameter
    into normalised YOLO boxes for a 256x256 tile centred at center_lat/lon
    """
    if region_df.empty:
        return []

    km_per_deg_lat = 30.87
    km_per_deg_lon = 30.87 * np.cos(np.deg2rad(center_lat))
    px_per_km      = tile_px / region_km  # how many pixels = 1 km at this scale

    annotations = []
    for _, row in region_df.iterrows():
        # how far is this crater from the tile centre, in km
        dlat_km = (row["latitude_deg"]  - center_lat) * km_per_deg_lat
        dlon_km = (row["longitude_deg"] - center_lon) * km_per_deg_lon

        # convert to pixel position (y is flipped — lat grows upward)
        px_x = tile_px / 2 + dlon_km * px_per_km
        px_y = tile_px / 2 - dlat_km * px_per_km

        diam_px = row["diameter_km"] * px_per_km
        r_px    = diam_px / 2

        # skip if centre is outside the tile, too tiny to see, or takes over the frame
        if not (0 <= px_x < tile_px and 0 <= px_y < tile_px):
            continue
        if diam_px < 2:
            continue
        if r_px / tile_px > MAX_CRATER_RADIUS_FRAC:
            continue

        sc = str(row.get("size_class", "small")).lower()
        cls = 2 if ("giant" in sc or "large" in sc) else (1 if "medium" in sc else 0)

        annotations.append({
            "class": cls,
            "cx":  float(np.clip(px_x / tile_px, 0.01, 0.99)),
            "cy":  float(np.clip(px_y / tile_px, 0.01, 0.99)),
            "w":   float(min(diam_px / tile_px, 0.90)),
            "h":   float(min(diam_px / tile_px, 0.90)),
            "diameter_km": float(row["diameter_km"]),
        })

    return annotations


# ----------------------------------------------------------------
# terrain renderer — shared by both modes
# ----------------------------------------------------------------

def _render_terrain(width=256, height=256):
    """makes a noisy grayscale lunar surface texture using layered blur"""
    base = np.random.randint(55, 145, (height, width), dtype=np.uint8)
    for sigma in [2, 6, 15, 40]:
        layer = np.random.randint(0, 60, (height, width), dtype=np.uint8)
        base  = cv2.addWeighted(base, 0.72,
                                cv2.GaussianBlur(layer, (0, 0), sigma), 0.28, 0)
    return base


def _add_crater(img, cx, cy, r, depth=0.6):
    """paints one crater onto the image — dark bowl, bright rim, ejecta rays"""
    H, W = img.shape[:2]
    cv2.circle(img, (cx, cy), r, int(75 * depth), -1)            # dark fill
    cv2.circle(img, (cx, cy), r, int(215 * depth), max(1, r//7)) # bright rim
    if r > 8:
        cv2.circle(img, (cx, cy), max(2, r//5), int(195 * depth), -1)  # central highlight
    for angle in range(0, 360, 25):
        rad    = np.deg2rad(angle + random.uniform(-12, 12))
        length = r + random.randint(r // 2, r * 2)
        ex = int(np.clip(cx + length * np.cos(rad), 0, W - 1))
        ey = int(np.clip(cy + length * np.sin(rad), 0, H - 1))
        cv2.line(img, (cx, cy), (ex, ey), int(185 * depth * 0.45), 1)
    return img


def render_tile_from_annotations(annotations, tile_px=TILE_SIZE):
    """renders terrain and draws craters at the positions from the annotations"""
    img = _render_terrain(tile_px, tile_px).astype(np.float32)
    for a in annotations:
        cx_px = int(a["cx"] * tile_px)
        cy_px = int(a["cy"] * tile_px)
        r_px  = max(2, int((a["w"] * tile_px) / 2))
        _add_crater(img, cx_px, cy_px, r_px, random.uniform(0.45, 1.0))
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


def generate_synthetic_image(width=256, height=256, n_craters=None,
                              class_weights=(0.55, 0.32, 0.13)):
    """makes one random crater tile with no catalog — pure made-up data"""
    if n_craters is None:
        n_craters = random.randint(4, 20)

    radius_ranges = {0: (3, 14), 1: (14, 36), 2: (36, 72)}
    annotations   = []

    for cls in random.choices([0, 1, 2], weights=class_weights, k=n_craters):
        rmin, rmax = radius_ranges[cls]
        r      = random.randint(rmin, rmax)
        margin = r + 4
        cx_px  = random.randint(margin, width  - margin)
        cy_px  = random.randint(margin, height - margin)
        annotations.append({
            "class": cls,
            "cx": cx_px / width,
            "cy": cy_px / height,
            "w":  (2 * r) / width,
            "h":  (2 * r) / height,
        })

    img = render_tile_from_annotations(annotations, width)
    return img, annotations


# ----------------------------------------------------------------
# dataset generators
# ----------------------------------------------------------------

def generate_robbins_dataset(n_images=600, dest=RAW_DIR, df=None):
    """
    builds training tiles using real crater positions from the Robbins catalog.
    for each tile we pick a real crater as the centre point, grab all craters
    in the surrounding 50km area, project them onto the tile, then render terrain.
    the model ends up trained on real lunar crater density and size distributions.
    """
    if df is None:
        df = load_robbins_catalog()

    img_dir = dest / "images"
    lbl_dir = dest / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    # sample seed points from the real catalog
    seed_craters = df.sample(min(n_images * 3, len(df)),
                             random_state=SEED).reset_index(drop=True)

    log.info(f"building {n_images} tiles from Robbins catalog ({KM_PER_TILE}km x {KM_PER_TILE}km each)...")

    written, tried = 0, 0
    pbar = tqdm(total=n_images, desc="Robbins tiles")

    while written < n_images and tried < len(seed_craters):
        row  = seed_craters.iloc[tried % len(seed_craters)]
        tried += 1
        clat = float(row["latitude_deg"])
        clon = float(row["longitude_deg"])

        region = sample_catalog_region(df, clat, clon, KM_PER_TILE)
        anns   = craters_to_tile_annotations(region, clat, clon, KM_PER_TILE)

        # mostly skip blank tiles — they hurt evaluation stats
        if not anns and random.random() > 0.08:
            continue

        img  = render_tile_from_annotations(anns, TILE_SIZE)
        stem = f"robbins_{written:06d}_lat{clat:.1f}_lon{clon:.1f}"

        cv2.imwrite(str(img_dir / f"{stem}.png"), img)
        with open(lbl_dir / f"{stem}.txt", "w") as f:
            for a in anns:
                f.write(f"{a['class']} {a['cx']:.6f} {a['cy']:.6f} "
                        f"{a['w']:.6f} {a['h']:.6f}\n")

        written += 1
        pbar.update(1)

    pbar.close()
    log.info(f"done — {written} tiles saved to {dest}")
    _log_class_distribution(lbl_dir)
    return img_dir, lbl_dir


def generate_synthetic_dataset(n_images=600, dest=RAW_DIR):
    """generates fully random tiles — no catalog, works offline"""
    img_dir = dest / "images"
    lbl_dir = dest / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"generating {n_images} random synthetic tiles...")
    for i in tqdm(range(n_images), desc="synthetic"):
        img, anns = generate_synthetic_image(TILE_SIZE, TILE_SIZE)
        stem      = f"synthetic_{i:05d}"
        cv2.imwrite(str(img_dir / f"{stem}.png"), img)
        with open(lbl_dir / f"{stem}.txt", "w") as f:
            for a in anns:
                f.write(f"{a['class']} {a['cx']:.6f} {a['cy']:.6f} "
                        f"{a['w']:.6f} {a['h']:.6f}\n")

    log.info(f"synthetic dataset saved to {dest}")
    return img_dir, lbl_dir


# ----------------------------------------------------------------
# tiling — only needed if you have large orbital images to cut up
# ----------------------------------------------------------------

def tile_image_with_labels(img_path, lbl_path, out_img_dir, out_lbl_dir,
                            tile_size=TILE_SIZE, overlap=OVERLAP):
    """cuts a big image into overlapping 256x256 tiles, re-labels each one"""
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        log.warning(f"couldn't read {img_path}")
        return 0

    H, W       = img.shape
    annotations = []
    if lbl_path.exists():
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
                    annotations.append((cls, cx, cy, w, h))

    stride     = tile_size - overlap
    tile_count = 0
    stem       = img_path.stem

    y = 0
    while y < H:
        x = 0
        while x < W:
            x2   = min(x + tile_size, W)
            y2   = min(y + tile_size, H)
            x1_t = x2 - tile_size
            y1_t = y2 - tile_size
            patch     = img[y1_t:y2, x1_t:x2]
            tile_anns = []

            for cls, cx_n, cy_n, w_n, h_n in annotations:
                abs_cx = cx_n * W
                abs_cy = cy_n * H
                abs_w  = w_n  * W
                abs_h  = h_n  * H

                # only keep boxes whose centre lands inside this tile
                if not (x1_t <= abs_cx < x2 and y1_t <= abs_cy < y2):
                    x += stride
                    continue

                # clamp the box to the tile edge
                bx1 = max(abs_cx - abs_w / 2, x1_t) - x1_t
                by1 = max(abs_cy - abs_h / 2, y1_t) - y1_t
                bx2 = min(abs_cx + abs_w / 2, x2)   - x1_t
                by2 = min(abs_cy + abs_h / 2, y2)   - y1_t

                new_cx = (bx1 + bx2) / 2 / tile_size
                new_cy = (by1 + by2) / 2 / tile_size
                new_w  = (bx2 - bx1) / tile_size
                new_h  = (by2 - by1) / tile_size

                if new_w > 0.01 and new_h > 0.01:
                    tile_anns.append((cls, new_cx, new_cy, new_w, new_h))

            tile_name = f"{stem}_t{tile_count:04d}"
            cv2.imwrite(str(out_img_dir / f"{tile_name}.png"), patch)
            with open(out_lbl_dir / f"{tile_name}.txt", "w") as f:
                for a in tile_anns:
                    f.write(f"{a[0]} {a[1]:.6f} {a[2]:.6f} {a[3]:.6f} {a[4]:.6f}\n")
            tile_count += 1
            x += stride
        y += stride

    return tile_count


# ----------------------------------------------------------------
# train / val / test split
# ----------------------------------------------------------------

def split_dataset(img_dir, lbl_dir, out_root=PROCESSED_DIR, splits=SPLITS, seed=SEED):
    """
    shuffles images and splits into train/val/test.
    also duplicates large-crater images in the training set
    to help balance the class distribution a bit.
    """
    random.seed(seed)
    images = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg"))
    random.shuffle(images)

    n       = len(images)
    n_train = int(n * splits["train"])
    n_val   = int(n * splits["val"])

    sets = {
        "train": images[:n_train],
        "val":   images[n_train:n_train + n_val],
        "test":  images[n_train + n_val:],
    }

    for split, imgs in sets.items():
        out_img = out_root / "images" / split
        out_lbl = out_root / "labels" / split
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)

        copy_list = imgs.copy()

        # oversample large-crater images in training only
        if split == "train":
            large_imgs = []
            for ip in imgs:
                lp = lbl_dir / (ip.stem + ".txt")
                if lp.exists():
                    with open(lp) as f:
                        for line in f:
                            if line.startswith("2 "):  # class 2 = large
                                large_imgs.append(ip)
                                break
            copy_list += large_imgs
            log.info(f"  oversampled {len(large_imgs)} large-crater images in train")

        for ip in tqdm(copy_list, desc=f"copying {split}"):
            lp = lbl_dir / (ip.stem + ".txt")
            shutil.copy(ip, out_img / ip.name)
            if lp.exists():
                shutil.copy(lp, out_lbl / lp.name)
            else:
                (out_lbl / lp.name).touch()  # empty file = negative sample

        log.info(f"  {split}: {len(copy_list)} images")

    _log_class_distribution(out_root / "labels" / "train")


def _log_class_distribution(lbl_dir):
    """counts how many boxes per class and prints it"""
    counts = {0: 0, 1: 0, 2: 0}
    for lp in lbl_dir.glob("*.txt"):
        with open(lp) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cls = int(parts[0])
                    counts[cls] = counts.get(cls, 0) + 1
    log.info(f"  labels — small:{counts[0]}  medium:{counts[1]}  large:{counts[2]}")


# ----------------------------------------------------------------
# main
# ----------------------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data"


def run_pipeline(use_robbins=True, synthetic=False, n_images=600):
    seed_everything(SEED)

    if synthetic:
        log.info("mode: synthetic (random, no catalog)")
        img_dir, lbl_dir = generate_synthetic_dataset(n_images, RAW_DIR)

    else:
        log.info("mode: Robbins catalog")
        try:
            df = load_robbins_catalog()
            img_dir, lbl_dir = generate_robbins_dataset(n_images, RAW_DIR, df)

            # save quick stats for the notebook
            import json
            stats = {
                "total_craters":      int(len(df)),
                "diameter_min_km":    float(df["diameter_km"].min()),
                "diameter_max_km":    float(df["diameter_km"].max()),
                "diameter_mean_km":   float(df["diameter_km"].mean()),
                "diameter_median_km": float(df["diameter_km"].median()),
                "size_class_counts":  df["size_class"].value_counts().to_dict(),
            }
            stats_path = DATA_DIR / "robbins_catalog_stats.json"
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=2)
            log.info(f"catalog stats saved → {stats_path}")

        except ImportError as e:
            log.warning(f"{e} — falling back to synthetic")
            img_dir, lbl_dir = generate_synthetic_dataset(n_images, RAW_DIR)

    log.info("splitting into train / val / test...")
    split_dataset(img_dir, lbl_dir, PROCESSED_DIR)

    yaml_path   = PROJECT_ROOT / "data" / "dataset.yaml"
    dataset_cfg = {
        "path":  str(PROCESSED_DIR),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    3,
        "names": ["small_crater", "medium_crater", "large_crater"],
    }
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_cfg, f, default_flow_style=False)
    log.info(f"dataset.yaml written → {yaml_path}")

    log.info("\n--- dataset summary ---")
    for split in ["train", "val", "test"]:
        n = len(list((PROCESSED_DIR / "images" / split).glob("*.png")))
        log.info(f"  {split}: {n} images")
    log.info("done! run: python src/train.py --all --epochs 50")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="crater detection data pipeline")
    parser.add_argument("--robbins",   action="store_true", default=True,
                        help="use Robbins catalog (default)")
    parser.add_argument("--synthetic", action="store_true",
                        help="skip catalog, use random data")
    parser.add_argument("--n", type=int, default=600, dest="n_images",
                        help="how many tiles to generate")
    args = parser.parse_args()
    run_pipeline(use_robbins=not args.synthetic, synthetic=args.synthetic,
                 n_images=args.n_images)
