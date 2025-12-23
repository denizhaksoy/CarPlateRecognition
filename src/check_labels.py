import re
import pandas as pd

def clean(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())

df = pd.read_csv("data/ocr_dataset.csv")
df = df[df["split"] == "train"].head(20).copy()

ok = 0
for _, r in df.iterrows():
    img_path = str(r["image_path"])
    # hem / hem \ için güvenli
    fn = img_path.replace("\\", "/").split("/")[-1]
    stem = fn.rsplit(".", 1)[0]

    gt_from_name = clean(stem.split("_")[0])   # dosya adının başı
    gt_from_csv  = clean(r["text"])            # CSV text

    print(gt_from_name, gt_from_csv, fn)
    ok += int(gt_from_name == gt_from_csv)

print(f"\nMATCHED {ok} of {len(df)}")
