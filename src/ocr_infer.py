from pathlib import Path
import argparse
import re
from PIL import ImageOps
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


def clean_plate(s: str) -> str:
    # Plaka için basit temizlik: sadece A-Z0-9 bırak
    s = s.upper().strip()
    s = re.sub(r"[^A-Z0-9]", "", s)
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, default="runs_ocr/trocr_poc/checkpoint-200")
    ap.add_argument("--image", type=str, required=True)
    ap.add_argument("--beam", type=int, default=1)
    args = ap.parse_args()

    device = torch.device("cpu")

    model_dir = args.model_dir
    base_name = "microsoft/trocr-small-printed"

    processor = TrOCRProcessor.from_pretrained(base_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir).to(device)
    model.eval()

    img_path = Path(args.image)
    image = Image.open(img_path).convert("RGB")
    import cv2
    import numpy as np

    img_np = np.array(image)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    image = Image.fromarray(gray).convert("RGB")
    # OCR-friendly preprocessing
    image = ImageOps.autocontrast(image)
    image = image.resize((image.size[0] * 2, image.size[1] * 2), Image.BICUBIC)

    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values,
            num_beams=8,
            min_length=7,
            max_length=7,
            length_penalty=2.0,
            early_stopping=True,
        )

    pred = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    pred_clean = clean_plate(pred)

    # GT label dosya isminden geliyor: PLATE_....
    gt = img_path.stem.split("_")[0]
    gt_clean = clean_plate(gt)

    print(f"IMG: {img_path.as_posix()}")
    print(f"GT : {gt} (clean={gt_clean})")
    print(f"PRD: {pred} (clean={pred_clean})")

if __name__ == "__main__":
    main()
