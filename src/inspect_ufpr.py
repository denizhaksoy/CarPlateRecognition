from pathlib import Path

RAW_ROOT = Path("data/raw/UFPR-ALPR")  

def main():
    count = 0
    for img_path in RAW_ROOT.rglob("*"):
        if img_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            txt_path = img_path.with_suffix(".txt")
            if not txt_path.exists():
                continue

            print("=" * 80)
            print("IMAGE:", img_path)
            print("ANN  :", txt_path)

            with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f):
                    print(f"  {i:02d}: {line.rstrip()}")
                    if i >= 7: 
                        break

            count += 1
            if count >= 3:  
                break

    if count == 0:
        print("0")

if __name__ == "__main__":
    main()
