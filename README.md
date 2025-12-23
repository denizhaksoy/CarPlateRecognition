# Automatic License Plate Recognition (ALPR)

This repository contains an end-to-end **Automatic License Plate Recognition (ALPR)** system developed as an **academic project**.

The system follows a two-stage deep learning pipeline:
1. **License Plate Detection** using YOLO  
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

> Note: The raw dataset is not included in this repository due to size constraints.

---

## System Architecture

Pipeline overview:

```
Raw Image
 → YOLO Plate Detection
 → Plate Cropping
 → OCR Dataset (CSV)
 → TrOCR Training / Inference
 → Text Prediction & Evaluation
```

---

## Repository Structure

```
.
├── src/                     Source code (training, inference, evaluation)
├── samples/
│   └── inputs/              Sample license plate images (qualitative examples)
├── results/
│   ├── preds.csv            OCR predictions for sample images
│   └── eval_summary.txt     Short evaluation summary
├── runs/
│   └── detect/train/        YOLO training outputs
├── data.yaml                YOLO dataset configuration
├── README.md
└── .gitignore
```

Large training artifacts are intentionally excluded (see below).

---

## License Plate Detection (YOLO)

YOLO is trained as a **single-class detector** (license plate).
Bounding boxes are derived from UFPR corner annotations.

**Training**
```bash
python src/train_yolo.py
```

**Output**
```
runs/detect/train/weights/best.pt
```

---

## Plate Cropping

Detected plates are cropped from original images using YOLO predictions and saved by dataset split.

```bash
python src/crop_plates.py
```

---

## OCR Dataset Construction

Cropped plates are linked with ground-truth text in a CSV file.

**CSV format**
```
image_path,text,split
```

---

## OCR Model (TrOCR)

**Base model**
```
microsoft/trocr-small-printed
```

**Training**
```bash
python src/train_trocr.py
```

Training produces large checkpoint files which are **not included** in this repository.

---

## OCR Inference

OCR inference is performed on a **single image**:

```bash
python src/ocr_infer.py --model_dir PATH_TO_MODEL --image PATH_TO_IMAGE --beam 5
```

---

## Evaluation

The OCR module is evaluated using:
- Exact Match
- Normalized Exact Match
- Character Error Rate (CER)

```bash
python src/eval_ocr.py --model_dir PATH_TO_MODEL --split test
```

---

## Results (Evidence for Evaluation)

Due to GitHub size limitations, **full training artifacts (checkpoints, optimizer states)** are excluded.

Instead, this repository provides **verifiable evidence** through:
- **Sample input images**: `samples/inputs/`
- **OCR predictions**: `results/preds.csv`
- **Evaluation summary**: `results/eval_summary.txt`

This allows instructors to verify that the model was trained and produces valid outputs,
while keeping the repository lightweight and manageable.

---

## Reproducibility

All scripts required to:
- train the detector
- crop license plates
- build the OCR dataset
- train and evaluate TrOCR  

are included in the `src/` directory.

Given the same dataset and configurations, the results can be reproduced.

---

## Limitations

- OCR accuracy is affected by low-resolution plate crops
- Strong character ambiguity (e.g., O/0, B/8)
- CPU-only training constraints
- Limited training data

---

## License

This repository is intended for **academic and educational use only**.
