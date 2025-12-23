# Automatic License Plate Recognition (ALPR)

This repository contains an end-to-end **Automatic License Plate Recognition (ALPR)** system developed as an **academic project**.

The system follows a two-stage deep learning pipeline:
1. **License Plate Detection** using YOLOv8
2. **License Plate Text Recognition (OCR)** using TrOCR

The project is designed to be **modular, reproducible, and suitable for academic evaluation**.

---

## Project Overview

Automatic License Plate Recognition (ALPR) is a core task in intelligent transportation systems.
This project implements a complete ALPR pipeline that operates under challenging real-world conditions such as:
- motion blur
- low resolution
- character ambiguity
- real driving scenarios

The **UFPR-ALPR dataset** is used due to its realistic nature and high difficulty.

---

## Dataset

**UFPR-ALPR Dataset**
- Real driving sequences captured from a moving vehicle
- License plate annotations and ground-truth text
- Predefined **training / validation / testing** splits

The original dataset splits are preserved to **avoid data leakage**.

Note: The raw dataset is not included in this repository due to size constraints.

---

## System Architecture

Raw Image
→ YOLOv8 License Plate Detection
→ Plate Cropping
→ OCR Dataset (CSV)
→ TrOCR Training / Inference
→ Text Prediction & Evaluation

---

## Repository Structure

.
├── src/                     Source code (training, inference, evaluation)
├── samples/
│   └── inputs/              Sample license plate images (qualitative examples)
├── results/
│   ├── preds.csv            OCR predictions for sample images
│   └── eval_summary.txt     Evaluation summary and detection metrics
├── runs/
│   └── detect/train/        YOLOv8 training outputs
├── data.yaml                YOLO dataset configuration
├── yolov8n.pt               Pretrained YOLOv8n weights
├── README.md
└── .gitignore

Large training artifacts are intentionally excluded.

---

## License Plate Detection (YOLOv8)

License plate detection is performed using the **Ultralytics YOLOv8 framework**.
A pretrained **YOLOv8n** model is fine-tuned as a **single-class detector** (license plate).

Bounding box annotations are generated from corner coordinates provided by the UFPR-ALPR dataset
and converted to YOLO format.

Training:
python src/train_yolo.py

Output:
runs/detect/train/weights/best.pt

---

## Detection Results

Precision: 0.91
Recall: 0.88
mAP@0.5: 0.95
mAP@0.5:0.95: 0.56

---

## Plate Cropping

python src/crop_plates.py

---

## OCR Dataset Construction

CSV format:
image_path,text,split

---

## OCR Model (TrOCR)

Base model:
microsoft/trocr-small-printed

Training:
python src/train_trocr.py

Large checkpoints are excluded due to GitHub size limits.

---

## OCR Inference

python src/ocr_infer.py --model_dir PATH_TO_MODEL --image PATH_TO_IMAGE --beam 5

---

## Evaluation

Exact Match Accuracy
Normalized Exact Match Accuracy
Character Error Rate (CER)

python src/eval_ocr.py --model_dir PATH_TO_MODEL --split test

---

## Results and Evidence

- Sample input images: samples/inputs/
- OCR predictions: results/preds.csv
- Evaluation summary: results/eval_summary.txt

---

## Reproducibility

All scripts required for training, inference, and evaluation are included in src/.

---

## Limitations

- Low-resolution plate crops
- Character ambiguity
- CPU-only training
- Limited dataset size

---

## License

Academic and educational use only.
