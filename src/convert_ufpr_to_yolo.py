import os
import shutil
from pathlib import Path
from PIL import Image

RAW_ROOT = Path("data/raw/UFPR-ALPR")
YOLO_ROOT = Path("data/yolo")

def parse_annotation(txt_path, img_width, img_height):
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [l.strip() for l in f if l.strip()]

    corner_line = None
    for line in lines:
        if "corners:" in line:
            corner_line = line
            break

    if corner_line is None:
        print(f"[WARN]: {txt_path}")
        return None

    try:
        after = corner_line.split("corners:")[1].strip()
        parts = after.split()

        xs, ys = [], []
        for p in parts:
            x_str, y_str = p.split(",")
            xs.append(float(x_str))
            ys.append(float(y_str))

        x_min = min(xs)
        x_max = max(xs)
        y_min = min(ys)
        y_max = max(ys)
        x_center = (x_min + x_max) / 2.0 / img_width
        y_center = (y_min + y_max) / 2.0 / img_height
        w = (x_max - x_min) / img_width
        h = (y_max - y_min) / img_height

        return 0, x_center, y_center, w, h  

    except Exception as e:
        print(f"[ERROR]: {txt_path} -> {e}")
        return None


def setup_yolo_dirs():
    print(f"[INFO] YOLO_ROOT = {YOLO_ROOT.resolve()}")
    if YOLO_ROOT.exists():
        shutil.rmtree(YOLO_ROOT)

    for split in ["train", "val", "test"]:
        (YOLO_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)


def convert_split(split_name, raw_subdir):
    src_root = RAW_ROOT / raw_subdir
    dst_img_root = YOLO_ROOT / "images" / split_name
    dst_lbl_root = YOLO_ROOT / "labels" / split_name

    print(f" {src_root.resolve()}")

    if not src_root.exists():
        return

    count = 0
    for img_path in src_root.rglob("*.png"):
        txt_path = img_path.with_suffix(".txt")
        if not txt_path.exists():
            continue

        try:
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception as e:
            continue

        parsed = parse_annotation(txt_path, w, h)
        if parsed is None:
            continue

        class_id, xc, yc, bw, bh = parsed

        dst_img = dst_img_root / img_path.name
        dst_lbl = dst_lbl_root / (img_path.stem + ".txt")

        shutil.copy2(img_path, dst_img)

        with open(dst_lbl, "w", encoding="utf-8") as f:
            f.write(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

        count += 1



def main():
    print(f"[INFO] RAW_ROOT = {RAW_ROOT.resolve()}")
    print(f"[INFO] RAW_ROOT var mı? {RAW_ROOT.exists()}")

    setup_yolo_dirs()

    convert_split("train", "training")
    convert_split("val",   "validation")
    convert_split("test",  "testing")


if __name__ == "__main__":
    main()
    
