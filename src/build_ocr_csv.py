from pathlib import Path 
import csv

CROPS_ROOT = Path("data/crops")

SPLITS = {
    "training": "train",
    "validation": "val",
    "testing": "test",
}

OUT_PATH = Path("data/ocr_dataset.csv")


def main():
    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "image_path", "text"])

        for ufpr_split, split_name in SPLITS.items():
            split_dir = CROPS_ROOT / ufpr_split
            if not split_dir.exists():
                print(f"[WARN] Split folder not found: {split_dir}")
                continue

            count = 0
            for img_path in split_dir.glob("*.png"):
                stem = img_path.stem
                plate_text = stem.split("_")[0] 

                writer.writerow([
                    split_name,              
                    img_path.as_posix(),   
                    plate_text,              
                ])
                count += 1

            print(f"[INFO] {ufpr_split}: {count} crops")

if __name__ == "__main__":
    main()
