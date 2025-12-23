from pathlib import Path
import random
import re
import pandas as pd
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

def clean(s: str) -> str:
    s = s.upper().strip()
    return re.sub(r"[^A-Z0-9]", "", s)

def main():
    model_dir = "runs_ocr/trocr_poc/final"
    base_name = "microsoft/trocr-small-printed"

    device = torch.device("cpu")
    processor = TrOCRProcessor.from_pretrained(model_dir) if (Path(model_dir)/"preprocessor_config.json").exists() else TrOCRProcessor.from_pretrained(base_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir).to(device)
    model.eval()

    files = list(Path("data/crops/testing").glob("*.png"))
    random.shuffle(files)
    files = files[:20]

    rows = []
    for p in files:
        gt = clean(p.stem.split("_")[0])
        image = Image.open(p).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            ids = model.generate(pixel_values, max_length=16, num_beams=1)
        pred = clean(processor.batch_decode(ids, skip_special_tokens=True)[0])
        rows.append({"file": p.name, "gt": gt, "pred": pred, "ok": (gt == pred)})

    df = pd.DataFrame(rows)
    print(df[["gt","pred","ok","file"]].to_string(index=False))
    print("\nExact match:", df["ok"].mean())

if __name__ == "__main__":
    main()
