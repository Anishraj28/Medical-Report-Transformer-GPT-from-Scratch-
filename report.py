# vision_report_training.py
import os
import math
import random
from pathlib import Path
from typing import List, Dict, Optional

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm

from transformers import (
    ViTImageProcessor, AutoTokenizer,
    VisionEncoderDecoderModel, get_scheduler
)


# ---------------------------
# CONFIG
# ---------------------------
CSV_PATH = "/Users/anishrajumapathy/Downloads/archive-4/indiana_reports.csv"        # path to your CSV
IMAGES_DIR = "/Users/anishrajumapathy/Downloads/archive-4/images/images_normalized"        # folder with PNG images
OUTPUT_DIR = "saved_report_model"       # where to save model & tokenizer
IMPRESSION_COL = "impression"           # column name to use as target text
UID_COL = "uid"                         # column with identifier to match images
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# training
BATCH_SIZE = 8
EPOCHS = 20
LR = 0.005
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 100
MAX_TARGET_LENGTH = 128   # max tokens for report
MAX_PIXEL_VALUE = 1.0

# generation settings
GEN_MAX_LENGTH = 128
NUM_BEAMS = 4

# small debug overfit mode (set to True to quickly verify training correctness)
DEBUG_OVERFIT = False
OVERFIT_SIZE = 32


# ---------------------------
# Utilities: robust filename matching
# ---------------------------
def find_image_files_for_uid(images_dir: str, uid: str) -> List[str]:
    """
    Return list of filenames in images_dir that contain uid as substring and end with .png
    This is intentionally permissive to handle formatting differences.
    """
    uid_str = str(uid)
    matches = []
    for fname in os.listdir(images_dir):
        if not fname.lower().endswith(".png"):
            continue
        if uid_str in fname:
            matches.append(os.path.join(images_dir, fname))
    return matches


# ---------------------------
# Dataset
# ---------------------------
class IndianaReportDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        images_dir: str,
        image_processor: ViTImageProcessor,
        tokenizer,
        uid_col: str = UID_COL,
        text_col: str = IMPRESSION_COL,
        max_target_length: int = MAX_TARGET_LENGTH,
        transform=None,
        require_image: bool = True
    ):
        self.df = pd.read_csv(csv_path)
        # Drop rows missing impression
        self.df = self.df.dropna(subset=[text_col])
        self.images_dir = images_dir
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.text_col = text_col
        self.uid_col = uid_col
        self.max_target_length = max_target_length
        self.transform = transform

        # Map rows -> first matched image (robust)
        rows = []
        for i, row in self.df.iterrows():
            uid = row[uid_col]
            matches = find_image_files_for_uid(images_dir, uid)
            if len(matches) == 0:
                # skip if no image found
                continue
            # prefer files that start with uid, else take first
            chosen = None
            for m in matches:
                if os.path.basename(m).startswith(str(uid)):
                    chosen = m
                    break
            if chosen is None:
                chosen = matches[0]
            rows.append({"img_path": chosen, "text": row[text_col], "uid": uid})

        self.rows = pd.DataFrame(rows).reset_index(drop=True)
        print(f"[DATA] total paired examples: {len(self.rows)}")

        # Quick sanity prints
        for i in range(min(5, len(self.rows))):
            print(f"[DATA CHECK] idx={i} uid={self.rows.loc[i,'uid']} img={os.path.basename(self.rows.loc[i,'img_path'])} text_sample={self.rows.loc[i,'text'][:80]!r}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows.iloc[idx]
        img_path = row["img_path"]
        text = str(row["text"])

        # Load image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Process image to model pixel_values
        pixel_values = self.image_processor(images=image, return_tensors="pt").pixel_values.squeeze(0)  # (C,H,W)

        # Tokenize text
        tokens = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_target_length,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.squeeze(0)
        attention_mask = tokens.attention_mask.squeeze(0)

        # Replace pad token id with -100 for loss ignoring if using labels directly
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "uid": row["uid"]
        }


# ---------------------------
# Collate function
# ---------------------------
def collate_fn(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    uids = [b["uid"] for b in batch]
    return {"pixel_values": pixel_values, "labels": labels, "input_ids": input_ids, "attention_mask": attention_mask, "uids": uids}


# ---------------------------
# Train / Eval / Generation functions
# ---------------------------
def train_loop(model, dataloader, optimizer, lr_scheduler, epoch, device):
    model.train()
    running_loss = 0.0
    for step, batch in enumerate(tqdm(dataloader, desc=f"Train E{epoch}")):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        if lr_scheduler:
            lr_scheduler.step()
        optimizer.zero_grad()

        running_loss += loss.item()
        if (step + 1) % 50 == 0:
            avg = running_loss / (step + 1)
            print(f"  step {step+1} loss {avg:.4f}")
    return running_loss / (len(dataloader) + 1e-12)


def eval_loop(model, dataloader, device, tokenizer, n_samples=10):
    model.eval()
    losses = []
    examples = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval"):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(pixel_values=pixel_values, labels=labels)
            losses.append(outputs.loss.item())

            # sample a handful for inspection
            if len(examples) < n_samples:
                gen = model.generate(
                    pixel_values=pixel_values[:1].to(device),
                    max_length=GEN_MAX_LENGTH,
                    num_beams=NUM_BEAMS,
                    early_stopping=True
                )
                decoded = tokenizer.decode(gen[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                examples.append((batch["uids"][0], decoded, tokenizer.decode(labels[0].masked_fill(labels[0]==-100, tokenizer.pad_token_id), skip_special_tokens=True)))

    avg_loss = sum(losses) / (len(losses) + 1e-12)
    return avg_loss, examples


def generate_report_from_image(model, processor, tokenizer, image_path, device, num_beams=NUM_BEAMS, max_length=GEN_MAX_LENGTH):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        output_ids = model.generate(pixel_values=pixel_values, num_beams=num_beams, max_length=max_length, early_stopping=True)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)


# ---------------------------
# Main: prepare model & data, train
# ---------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # load processors & tokenizers
    print("[INFO] loading processors & tokenizer")
    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")  # GPT-2 tokenizer

    # Ensure special tokens exist (gpt2 lacks pad token). We'll add pad and bos/eos if missing.
    added = False
    special_tokens = {}
    if tokenizer.pad_token is None:
        special_tokens["pad_token"] = "<|pad|>"
    if tokenizer.bos_token is None:
        special_tokens["bos_token"] = ""
    if tokenizer.eos_token is None:
        special_tokens["eos_token"] = ""

    if len(special_tokens) > 0:
        tokenizer.add_special_tokens(special_tokens)
        added = True
        print(f"[INFO] added special tokens: {special_tokens}")

    # load encoder-decoder model
    print("[INFO] creating VisionEncoderDecoderModel (ViT encoder + GPT2 decoder)")
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        "google/vit-base-patch16-224-in21k", "gpt2"
    )

    # if tokenizer changed size, resize decoder embeddings
    if added:
        model.decoder.resize_token_embeddings(len(tokenizer))

    # tie special token ids into config
    model.config.decoder_start_token_id = tokenizer.bos_token_id or tokenizer.eos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # generation config
    model.config.max_length = GEN_MAX_LENGTH
    model.config.num_beams = NUM_BEAMS

    model.to(DEVICE)

    # Prepare dataset
    # use impression as target (clean shorter diagnostic text)
    if IMPRESSION_COL not in pd.read_csv(CSV_PATH).columns:
        raise ValueError(f"CSV does not contain column '{IMPRESSION_COL}' — please modify CSV_PATH or IMPRESSION_COL")

    print("[INFO] creating dataset")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    full_dataset = IndianaReportDataset(
        csv_path=CSV_PATH,
        images_dir=IMAGES_DIR,
        image_processor=image_processor,
        tokenizer=tokenizer,
        uid_col=UID_COL,
        text_col=IMPRESSION_COL,
        max_target_length=MAX_TARGET_LENGTH,
        transform=None  # use image_processor instead for pixel_values
    )

    # optional debug overfit
    if DEBUG_OVERFIT:
        small_idx = list(range(min(OVERFIT_SIZE, len(full_dataset))))
        small_df = full_dataset.rows.iloc[small_idx].reset_index(drop=True)
        # write small CSV to memory by creating small dataset object
        full_dataset.rows = small_df
        print(f"[DEBUG] Overfit mode enabled: training on {len(full_dataset)} examples")

    # split train / val
    n = len(full_dataset)
    indices = list(range(n))
    random.shuffle(indices)
    split = int(0.8 * n)
    train_idx, val_idx = indices[:split], indices[split:]

    train_rows = full_dataset.rows.iloc[train_idx].reset_index(drop=True)
    val_rows = full_dataset.rows.iloc[val_idx].reset_index(drop=True)

    # create lightweight datasets using the rows DataFrame
    def make_dataset_from_rows(rows_df):
        ds = IndianaReportDataset.__new__(IndianaReportDataset)
        # shallow copy fields required
        ds.rows = rows_df.reset_index(drop=True)
        ds.image_processor = image_processor
        ds.tokenizer = tokenizer
        ds.max_target_length = MAX_TARGET_LENGTH
        ds.images_dir = IMAGES_DIR
        ds.text_col = IMPRESSION_COL
        ds.uid_col = UID_COL
        ds.transform = None
        return ds

    train_ds = make_dataset_from_rows(train_rows)
    val_ds = make_dataset_from_rows(val_rows)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    num_training_steps = EPOCHS * len(train_loader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=num_training_steps
    )

    # training loop
    best_val_loss = float("inf")
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_loop(model, train_loader, optimizer, lr_scheduler, epoch, DEVICE)
        val_loss, val_examples = eval_loop(model, val_loader, DEVICE, tokenizer)

        print(f"Epoch {epoch} -> train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")
        for uid, pred, ref in val_examples:
            print(f"[EXAMPLE] uid={uid}")
            print("  PRED:", pred)
            print("  REF :", ref[:200])

        # save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            print(f"[SAVE] saved best model to {OUTPUT_DIR}")

    # Final quick generation demo on a random validation image
    sample_row = val_rows.sample(1).iloc[0]
    demo_img = sample_row["img_path"]
    print(f"\nDemo generation for sample image: {demo_img}")
    report = generate_report_from_image(model, image_processor, tokenizer, demo_img, DEVICE)
    print("Generated report:\n", report)


if __name__ == "__main__":
    main()
