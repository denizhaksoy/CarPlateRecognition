import argparse
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageFilter

import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


def parse_plate_and_corners(ann_path: Path):
    plate = None
    corners = None

    with open(ann_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if "plate:" in line:
                plate = line.split("plate:")[1].strip()
            if "corners:" in line:
                after = line.split("corners:")[1].strip()
                pts = []
                for p in after.split():
                    x, y = p.split(",")
                    pts.append([float(x), float(y)])
                if len(pts) == 4:
                    corners = np.array(pts, dtype=np.float32)

    return plate, corners


def order_points(pts: np.ndarray):
    # standard TL, TR, BR, BL ordering
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def rectify_plate_bgr(img_bgr: np.ndarray, corners: np.ndarray, out_w=240, out_h=80, pad=6):
    corners = order_points(corners)

    # pad destination (a bit of border helps OCR)
    dst = np.array([
        [pad, pad],
        [out_w - pad, pad],
        [out_w - pad, out_h - pad],
        [pad, out_h - pad]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(img_bgr, M, (out_w, out_h))
    return warped


def pil_preprocess(pil_img: Image.Image):
    pil_img = pil_img.convert("L")  # grayscale

    # upscale
    pil_img = pil_img.resize((pil_img.size[0] * 4, pil_img.size[1] * 4), Image.BICUBIC)

    # binarize (simple threshold)
    np_img = np.array(pil_img)
    _, th = cv2.threshold(np_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    pil_img = Image.fromarray(th).convert("RGB")

    return pil_img


import re

def normalize_plate(s: str) -> str:
    s = (s or "").upper()

    # common OCR confusions
    s = s.replace("O", "0")
    s = s.replace("I", "1")
    s = s.replace("L", "1")
    s = s.replace("Z", "2")
    s = s.replace("S", "5")
    s = s.replace("B", "8")
    s = s.replace("U", "1")

    # keep only A-Z0-9
    s = re.sub(r"[^A-Z0-9]", "", s)

    # enforce UFPR length (7 chars)
    if len(s) > 7:
        s = s[:7]
    elif len(s) < 7:
        s = s.ljust(7, "0")

    return s



def clean_plate(s: str):
    # keep only A-Z0-9
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--model_dir", default="microsoft/trocr-small-printed")
    ap.add_argument("--beams", type=int, default=8)
    args = ap.parse_args()

    img_path = Path(args.image)
    ann_path = img_path.with_suffix(".txt")

    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation not found: {ann_path}")

    gt_text, corners = parse_plate_and_corners(ann_path)
    gt_text = clean_plate(gt_text)

    if corners is None:
        raise ValueError("corners not found in annotation")

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise ValueError(f"Cannot read image: {img_path}")

    plate_bgr = rectify_plate_bgr(img_bgr, corners, out_w=320, out_h=96, pad=8)
    plate_rgb = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(plate_rgb)
    pil_img = pil_preprocess(pil_img)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = TrOCRProcessor.from_pretrained(args.model_dir)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_dir).to(device)
    model.eval()

    pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values.to(device)

    # IMPORTANT: fixed-length decoding for UFPR plates (7 chars)
    generated_ids = model.generate(
        pixel_values,
        num_beams=8,
        max_length=7,
        min_length=7,
        early_stopping=True,
    )

    pred = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    pred_clean = clean_plate(pred)
    pred_raw = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    pred_clean = normalize_plate(pred_raw)
    pred_norm = normalize_plate(pred_clean)

    print(f"IMG : {img_path}")
    print(f"GT  : {gt_text}")
    print(f"PRD : {pred_raw} | clean={pred_clean} | norm={pred_norm}")
    print(f"LEN : gt={len(gt_text)} pred_norm={len(pred_norm)}")






if __name__ == "__main__":
    main()
