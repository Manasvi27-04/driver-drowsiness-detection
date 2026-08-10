"""
data_preprocessing.py

Loads raw driver-behavior images, validates/cleans them, and splits them
into train/val/test directories in an ImageDataGenerator-friendly layout:

    output_dir/
        train/{alert,sleepy,yawning}/
        val/{alert,sleepy,yawning}/
        test/{alert,sleepy,yawning}/
"""

import argparse
import os
import shutil
import random
from pathlib import Path
from PIL import Image
from tqdm import tqdm

CLASSES = ["alert", "sleepy", "yawning"]
IMG_SIZE = (224, 224)


def is_valid_image(path: Path) -> bool:
    """Check the image opens and isn't corrupted/truncated."""
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def collect_images(class_dir: Path):
    valid_exts = {".jpg", ".jpeg", ".png"}
    return [p for p in class_dir.iterdir() if p.suffix.lower() in valid_exts]


def split_dataset(files, train_ratio=0.7, val_ratio=0.15, seed=42):
    random.Random(seed).shuffle(files)
    n = len(files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return {
        "train": files[:n_train],
        "val": files[n_train:n_train + n_val],
        "test": files[n_train + n_val:],
    }


def resize_and_save(src_path: Path, dst_path: Path):
    with Image.open(src_path) as img:
        img = img.convert("RGB").resize(IMG_SIZE)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst_path, quality=95)


def main(data_dir: str, output_dir: str):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    summary = {}

    for cls in CLASSES:
        class_dir = data_dir / cls
        if not class_dir.exists():
            print(f"[WARN] Missing class folder: {class_dir}")
            continue

        files = collect_images(class_dir)
        valid_files = [f for f in tqdm(files, desc=f"Validating {cls}") if is_valid_image(f)]
        print(f"{cls}: {len(files)} found, {len(valid_files)} valid")

        splits = split_dataset(valid_files)
        summary[cls] = {k: len(v) for k, v in splits.items()}

        for split_name, split_files in splits.items():
            for f in tqdm(split_files, desc=f"Copying {cls}/{split_name}"):
                dst = output_dir / split_name / cls / f.name
                resize_and_save(f, dst)

    print("\n=== Split Summary ===")
    for cls, counts in summary.items():
        print(f"{cls}: {counts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess and split driver drowsiness dataset")
    parser.add_argument("--data_dir", required=True, help="Path to raw data with alert/sleepy/yawning folders")
    parser.add_argument("--output_dir", required=True, help="Output path for train/val/test split")
    args = parser.parse_args()
    main(args.data_dir, args.output_dir)
