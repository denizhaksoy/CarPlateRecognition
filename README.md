# Automatic License Plate Recognition (ALPR)

This repository contains an end-to-end Automatic License Plate Recognition (ALPR) system developed as an academic project.
The system follows a two-stage deep learning pipeline:

1. License Plate Detection using YOLO
2. License Plate Text Recognition (OCR) using TrOCR

The project is designed to be modular, reproducible, and suitable for academic evaluation.

---

## Project Overview

Automatic License Plate Recognition (ALPR) is a core task in intelligent transportation systems.
This project implements a complete ALPR pipeline that operates under challenging real-world conditions such as motion blur,
low resolution, and character ambiguity.

The UFPR-ALPR dataset is used due to its realistic nature and difficulty.

---

## Dataset

UFPR-ALPR dataset:
- Real driving sequences captured from a moving vehicle
- License plate annotations and ground-truth text
- Predefined training / validation / testing splits

The original dataset splits are preserved to avoid data leakage.

---

## System Architecture

Pipeline:

Raw Image
→ YOLO Plate Detection
→ Plate Cropping
→ OCR Dataset (CSV)
→ TrOCR Training / Inference
→ Text Prediction & Evaluation

---

## Project Structure

data/
- raw/UFPR-ALPR/            Original dataset (not included)
- yolo/                     YOLO-formatted detection dataset
- crops/
  - training/
  - validation/
  - testing/
- ocr_dataset.csv           OCR dataset (image_path, text, split)

runs/
- detect/train/             YOLO training outputs (best.pt)

runs_ocr/
- trocr_poc/
  - checkpoint-*
  - final/                  Final OCR model and processor

src/
- convert_to_yolo.py        UFPR → YOLO annotation conversion
- train_yolo.py             YOLO plate detector training
- crop_plates.py            Plate cropping using YOLO predictions
- build_ocr_csv.py          OCR CSV construction
- train_trocr.py            TrOCR fine-tuning
- ocr_infer.py              OCR inference on a single image
- eval_ocr.py               OCR evaluation

---

## License Plate Detection (YOLO)

YOLO is trained as a single-class detector (license plate).
Bounding boxes are derived from UFPR corner annotations.

Training:
python src/train_yolo.py

Output:
runs/detect/train/weights/best.pt

---

## Plate Cropping

Detected plates are cropped from original images using YOLO predictions and saved by dataset split.

Cropping:
python src/crop_plates.py

Output:
data/crops/{training,validation,testing}

---

## OCR Dataset Construction

Cropped plates are linked to ground-truth text in a CSV file.

CSV format:
image_path,text,split

Generation:
python src/build_ocr_csv.py

Output:
data/ocr_dataset.csv

---

## OCR Model (TrOCR)

Base model:
microsoft/trocr-small-printed

Training:
python src/train_trocr.py

Final model:
runs_ocr/trocr_poc/final/

---

## OCR Inference

python src/ocr_infer.py --model_dir runs_ocr/trocr_poc/final --image path/to/image.png

---

## Evaluation

Metrics:
- Exact Match (7 characters)
- Normalized Exact Match (ambiguity-aware)
- Character Error Rate (CER)

Evaluation:
python src/eval_ocr.py --model_dir runs_ocr/trocr_poc/final --split test

---

## Results and Limitations

The pipeline works end-to-end but OCR accuracy is limited due to:
- low-quality crops
- strong character ambiguity
- domain mismatch
- CPU-only training constraints
---
