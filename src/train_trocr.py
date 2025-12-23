import os
import random
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps, ImageFilter

import torch
from datasets import Dataset
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# ---------------- CONFIG ----------------
CSV_PATH = "data/ocr_dataset.csv"
MODEL_NAME = "microsoft/trocr-small-printed"
OUTPUT_DIR = "runs_ocr/trocr_final"

MAX_TARGET_LEN = 16
TRAIN_MAX_ROWS = None   # tüm train (1766)
VAL_MAX_ROWS = None     # tüm val (725)

torch.set_num_threads(max(1, os.cpu_count() // 2))
# ----------------------------------------


def load_split(split: str, max_rows=None):
    df = pd.read_csv(CSV_PATH)
    df = df[df["split"] == split].copy()

    df["text"] = df["text"].astype(str).str.strip()

    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42)

    return df.reset_index(drop=True)


class OCRDataset(torch.utils.data.Dataset):
    def __init__(self, df, processor):
        self.df = df
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def preprocess(self, image: Image.Image) -> Image.Image:
        image = ImageOps.autocontrast(image)
        image = image.resize((image.size[0] * 2, image.size[1] * 2), Image.BICUBIC)
        image = image.filter(ImageFilter.SHARPEN)
        return image

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        image = self.preprocess(image)

        pixel_values = self.processor(
            images=image, return_tensors="pt"
        ).pixel_values.squeeze(0)

        labels = self.processor.tokenizer(
            row["text"],
            padding="max_length",
            max_length=MAX_TARGET_LEN,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }


def main():
    print("[INFO] Loading data...")
    train_df = load_split("train", TRAIN_MAX_ROWS)
    val_df = load_split("val", VAL_MAX_ROWS)

    processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

        # =======================
    # PLATE ALPHABET FIX
    # =======================
    plate_chars = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    processor.tokenizer.add_tokens(plate_chars)
    model.decoder.resize_token_embeddings(len(processor.tokenizer))

    # Required config
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id

    # Generation config (safe)
    model.generation_config.max_length = MAX_TARGET_LEN
    model.generation_config.min_length = 7
    model.generation_config.num_beams = 5
    model.generation_config.early_stopping = True


    train_ds = OCRDataset(train_df, processor)
    val_ds = OCRDataset(val_df, processor)

    args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=5,
    learning_rate=2e-5,

    eval_strategy="steps",
    eval_steps=500,

    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,

    logging_steps=50,

    predict_with_generate=True,
    generation_max_length=MAX_TARGET_LEN,

    report_to="none",
    fp16=False,
    dataloader_num_workers=0,
    )


    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=processor,
    )

    trainer.train()

    final_dir = Path(OUTPUT_DIR) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(final_dir, safe_serialization=True)
    processor.save_pretrained(final_dir)

    print(f"[DONE] Final model saved to: {final_dir.resolve()}")


if __name__ == "__main__":
    main()
