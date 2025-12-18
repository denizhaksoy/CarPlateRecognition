from pathlib import Path #data/ocr_dataset olusturan script
import csv

# Cropların bulunduğu ana klasör
CROPS_ROOT = Path("data/crops")

# UFPR split adlarını OCR split isimlerine map ediyoruz
SPLITS = {
    "training": "train",
    "validation": "val",
    "testing": "test",
}

# Çıkacak CSV dosyasının yolu
OUT_PATH = Path("data/ocr_dataset.csv")


def main():
    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Header satırı
        writer.writerow(["split", "image_path", "text"])

        for ufpr_split, split_name in SPLITS.items():
            split_dir = CROPS_ROOT / ufpr_split
            if not split_dir.exists():
                print(f"[WARN] Split folder not found: {split_dir}")
                continue

            count = 0
            # Her split altındaki tüm .png crop'ları tara
            for img_path in split_dir.glob("*.png"):
                # Örnek isim: AAW9529_track0020[01]_0.png
                stem = img_path.stem  # "AAW9529_track0020[01]_0"
                plate_text = stem.split("_")[0]  # "AAW9529"

                writer.writerow([
                    split_name,              # train / val / test
                    img_path.as_posix(),     # "data/crops/..." şeklinde path
                    plate_text,              # plaka stringi
                ])
                count += 1

            print(f"[INFO] {ufpr_split}: {count} crops eklendi.")

    print(f"[DONE] CSV kaydedildi: {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
