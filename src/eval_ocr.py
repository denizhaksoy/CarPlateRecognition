# src/eval_ocr.py
import re
import random
import argparse

import pandas as pd
import torch
from PIL import Image, ImageOps, ImageFilter
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

PLATE_LEN = 7  # UFPR-ALPR plates are 7 chars (senin CSV'de 7)

# Ambiguity mapping (OCR common confusions)
AMBIG = str.maketrans({
    "I": "1",
    "L": "1",
    "O": "0",
    "Q": "0",
    "D": "0",
    "S": "5",
    "Z": "2",
    "B": "8",
    "G": "6",
})


def clean(s: str) -> str:
    """Keep only A-Z0-9, uppercase."""
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def norm_plate(s: str) -> str:
    """
    Domain normalize:
    1) clean (A-Z0-9)
    2) ambiguity mapping (I->1, O->0, etc.)
    3) first 7 chars
    """
    s = clean(s)
    s = s.translate(AMBIG)
    return s[:PLATE_LEN]


def cer(ref: str, hyp: str) -> float:
    """Levenshtein distance / len(ref) on cleaned strings."""
    ref = clean(ref)
    hyp = clean(hyp)
    if len(ref) == 0:
        return 1.0 if len(hyp) > 0 else 0.0

    dp = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, start=1):
        prev = dp[0]
        dp[0] = i
        for j, hc in enumerate(hyp, start=1):
            cur = dp[j]
            cost = 0 if rc == hc else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur

    return dp[-1] / len(ref)


def preprocess(image: Image.Image, upscale: int = 2, sharpen: bool = True) -> Image.Image:
    """Simple preprocessing."""
    image = ImageOps.autocontrast(image)
    if upscale and upscale != 1:
        image = image.resize((image.size[0] * upscale, image.size[1] * upscale), Image.BICUBIC)
    if sharpen:
        image = image.filter(ImageFilter.SHARPEN)
    return image


def generate_plate(model, pixel_values, beams: int):
    """
    Try to generate ~7 chars. Use max_new_tokens when available.
    Fallback to max_length if needed.
    """
    # Many TrOCR configs work better with max_new_tokens
    try:
        return model.generate(
            pixel_values,
            num_beams=beams,
            max_new_tokens=PLATE_LEN,
            early_stopping=False,  # greedy/beam validliği için
        )
    except TypeError:
        # Older versions: no max_new_tokens
        return model.generate(
            pixel_values,
            num_beams=beams,
            max_length=PLATE_LEN + 2,  # tokenization overhead olabilir
            early_stopping=False,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--csv", type=str, default="data/ocr_dataset.csv")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--beams", type=int, default=4)
    ap.add_argument("--upscale", type=int, default=2)
    ap.add_argument("--no_sharpen", action="store_true")
    ap.add_argument("--print_k", type=int, default=10)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df["split"] == args.split].copy()
    df["text"] = df["text"].astype(str).str.strip()

    if len(df) == 0:
        raise SystemExit(f"No rows for split={args.split}. Check CSV path/split values.")

    random.seed(args.seed)
    idxs = list(range(len(df)))
    random.shuffle(idxs)
    idxs = idxs[: min(args.n, len(df))]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = TrOCRProcessor.from_pretrained(args.model_dir)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_dir).to(device)
    model.eval()

    exact_clean7 = 0
    exact_norm7 = 0
    cer_list = []
    pred_lens = []

    printed = 0
    for k in idxs:
        row = df.iloc[k]
        img_path = row["image_path"]
        gt_raw = row["text"]

        image = Image.open(img_path).convert("RGB")
        image = preprocess(image, upscale=args.upscale, sharpen=(not args.no_sharpen))

        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

        with torch.no_grad():
            out_ids = generate_plate(model, pixel_values, beams=args.beams)

        pred_raw = processor.batch_decode(out_ids, skip_special_tokens=True)[0]

        gt_c7 = clean(gt_raw)[:PLATE_LEN]
        pr_c7 = clean(pred_raw)[:PLATE_LEN]

        gt_n7 = norm_plate(gt_raw)
        pr_n7 = norm_plate(pred_raw)

        exact_clean7 += int(gt_c7 == pr_c7)
        exact_norm7 += int(gt_n7 == pr_n7)

        cer_list.append(cer(gt_raw, pred_raw))
        pred_lens.append(len(clean(pred_raw)))

        if printed < args.print_k:
            print(f"IMG: {img_path}")
            print(f"GT : {gt_c7} | norm={gt_n7}")
            print(f"PRD: {pr_c7} | norm={pr_n7}")
            print("-" * 40)
            printed += 1

    n = len(idxs)
    print("\n=== METRICS ===")
    print(f"split={args.split}  n={n}  beams={args.beams}")
    print(f"Exact Match (clean7): {exact_clean7/n:.3f} ({exact_clean7}/{n})")
    print(f"Exact Match (norm7) : {exact_norm7/n:.3f} ({exact_norm7}/{n})")
    print(f"Avg CER            : {sum(cer_list)/n:.3f}")
    print(f"Avg pred len(clean): {sum(pred_lens)/n:.2f}")


if __name__ == "__main__":
    main()
