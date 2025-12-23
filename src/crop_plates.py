from ultralytics import YOLO
from pathlib import Path
import cv2
import os
import re

MODEL_PATH = "runs_cpu/exp_main/weights/best.pt"   
RAW_ROOT = Path("data/raw/UFPR-ALPR")             
OUT_DIR = Path("data/crops")

OUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_plate_text(annotation_path: Path):
    with open(annotation_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:

            if "plate:" in line:
                plate = line.split("plate:")[1].strip()
                return plate
            m = re.search(r"\bplate:\s*([A-Za-z0-9\-]+)", line)
            if m:
                return m.group(1).replace("-", "").upper()
    return None


def crop_plate(img, box):
    x1, y1, x2, y2 = map(int, box)
    h, w = img.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))
    return img[y1:y2, x1:x2]


def process_split(split_name: str):
    split_root = RAW_ROOT / split_name

    if not split_root.exists():
        return

    model = YOLO(MODEL_PATH)

    img_count = 0
    crop_count = 0

    for img_path in split_root.rglob("*.png"):
        ann_path = img_path.with_suffix(".txt")
        if not ann_path.exists():
            continue

        plate_text = extract_plate_text(ann_path)
        if not plate_text:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        results = model(img, verbose=False)[0]

        if len(results.boxes) == 0:
            continue

        for i, box in enumerate(results.boxes.xyxy.cpu().numpy()): 
            crop = crop_plate(img, box)

            if crop.size == 0:
                continue

            split_out_dir = OUT_DIR / split_name
            split_out_dir.mkdir(parents=True, exist_ok=True)

            out_name = f"{plate_text}_{img_path.stem}_{i}.png"
            out_path = split_out_dir / out_name

            cv2.imwrite(str(out_path), crop)
            crop_count += 1

        img_count += 1
        if img_count % 50 == 0:
            print(f"[INFO] {img_count}, {crop_count}, ({split_name}).")


if __name__ == "__main__":
    print(f"[INFO] RAW_ROOT = {RAW_ROOT.resolve()}")
    print(f"[INFO] MODEL_PATH = {Path(MODEL_PATH).resolve()}")

    for split in ["training", "validation", "testing"]:
        process_split(split)
