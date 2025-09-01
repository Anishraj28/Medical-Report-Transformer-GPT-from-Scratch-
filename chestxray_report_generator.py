import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import os
from typing import List, Dict
from tqdm import tqdm

# -----------------------------
# Configuration
# -----------------------------
class CFG:
    reports_csv = "/Users/anishrajumapathy/Downloads/archive-4/indiana_reports.csv"
    images_csv = "/Users/anishrajumapathy/Downloads/archive-4/indiana_projections.csv"
    image_root = "/Users/anishrajumapathy/Downloads/archive-4/images/images_normalized"
    batch_size = 8
    context_length = 128
    model_dim = 256
    num_blocks = 4
    num_heads = 8
    lr = 3e-4
    train_steps = 100
    device = "mps" if torch.backends.mps.is_available() else "cpu"

cfg = CFG()

# -----------------------------
# Dataset & Tokenizer
# -----------------------------
class ChestXrayReportDataset(Dataset):
    def __init__(self, reports_csv: str, images_csv: str, image_root: str, tokenizer, max_len: int):
        reports_df = pd.read_csv(reports_csv)   # uid, findings, impression
        images_df = pd.read_csv(images_csv)     # uid, filename, projection

        # Merge on uid
        self.df = pd.merge(reports_df, images_df, on="uid")
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

        # Prepare reports: use findings if available, otherwise impression
        self.reports = []
        for _, row in self.df.iterrows():
            text = str(row['findings']) if pd.notna(row['findings']) and len(str(row['findings']).strip()) > 0 else str(row['impression'])
            self.reports.append(text.strip())

    def __len__(self):
        return len(self.reports)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_root, self.df.iloc[idx]['filename'])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        report = self.tokenizer.encode(self.reports[idx])
        # Pad or truncate to max_len
        if len(report) > self.max_len:
            report = report[:self.max_len]
        else:
            report = report + [0] * (self.max_len - len(report))  # pad with 0

        return image, torch.tensor(report, dtype=torch.long)




# Simple byte-level tokenizer
class SimpleTokenizer:
    def __init__(self):
        self.stoi = {}
        self.itos = {}
    
    def build_vocab(self, texts: List[str]):
        chars = sorted(list(set("".join(texts))))
        self.stoi = {ch:i for i,ch in enumerate(chars)}
        self.itos = {i:ch for i,ch in enumerate(chars)}

    def encode(self, text: str):
        return [self.stoi[ch] for ch in text if ch in self.stoi]

    def decode(self, tokens: List[int]):
        return "".join([self.itos[t] for t in tokens])

# -----------------------------
# Transformer Decoder (from scratch)
# -----------------------------
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, model_dim, context_length, num_blocks, num_heads):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, model_dim)
        self.pos_embeddings = nn.Embedding(context_length, model_dim)
        self.blocks = nn.ModuleList([TransformerBlock(model_dim, num_heads) for _ in range(num_blocks)])
        self.final_ln = nn.LayerNorm(model_dim)
        self.vocab_projection = nn.Linear(model_dim, vocab_size)
    
    def forward(self, x):
        B, T = x.shape
        token_embeds = self.token_embeddings(x)
        pos_embeds = self.pos_embeddings(torch.arange(T, device=x.device).unsqueeze(0).repeat(B,1))
        h = token_embeds + pos_embeds
        for block in self.blocks:
            h = block(h)
        h = self.final_ln(h)
        logits = self.vocab_projection(h)
        return logits

class TransformerBlock(nn.Module):
    def __init__(self, model_dim, num_heads):
        super().__init__()
        self.mhsa = MultiHeadedSelfAttention(model_dim, model_dim, num_heads)
        self.ln1 = nn.LayerNorm(model_dim)
        self.ff = FeedForward(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)

    def forward(self, x):
        x = x + self.mhsa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, embed_dim, attn_dim, num_heads):
        super().__init__()
        self.heads = nn.ModuleList([SingleHeadAttention(embed_dim, attn_dim // num_heads) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=2)

class SingleHeadAttention(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.k = nn.Linear(embed_dim, head_dim)
        self.q = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)

    def forward(self, x):
        B,T,D = x.shape
        k = self.k(x)
        q = self.q(x)
        v = self.v(x)
        scores = q @ k.transpose(1,2) / (D ** 0.5)
        mask = torch.tril(torch.ones(T,T, device=x.device)) == 0
        scores = scores.masked_fill(mask, float('-inf'))
        probs = F.softmax(scores, dim=2)
        return probs @ v

class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 4*dim),
            nn.ReLU(),
            nn.Linear(4*dim, dim)
        )
    def forward(self, x):
        return self.net(x)

# -----------------------------
# Full model: ResNet encoder + Transformer decoder
# -----------------------------
class ReportGenerator(nn.Module):
    def __init__(self, vocab_size, model_dim, context_length, num_blocks, num_heads):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        modules = list(resnet.children())[:-1]  # remove classifier
        self.cnn = nn.Sequential(*modules)
        for p in self.cnn.parameters():
            p.requires_grad = False
        self.fc = nn.Linear(2048, model_dim)
        self.decoder = TransformerDecoder(vocab_size, model_dim, context_length, num_blocks, num_heads)

    def forward(self, images, captions):
        B = images.shape[0]
        features = self.cnn(images).view(B, -1)
        features = self.fc(features).unsqueeze(1)  # B x 1 x D
        # prepend feature as first token embedding
        logits = self.decoder(captions)
        return logits

# -----------------------------
# Training loop
# -----------------------------
def train():
    # Read CSVs
    df_reports = pd.read_csv(cfg.reports_csv)
    df_images = pd.read_csv(cfg.images_csv)

    # Merge on UID
    df = pd.merge(df_images, df_reports, on="uid")

    # Build tokenizer
    reports = []
    for idx, row in df.iterrows():
        text = str(row['findings']) if pd.notna(row['findings']) and len(str(row['findings']).strip()) > 0 else str(row['impression'])
        reports.append(text.strip())

    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(reports)
    vocab_size = len(tokenizer.stoi)

    # Dataset and DataLoader
    dataset = ChestXrayReportDataset(cfg.reports_csv, cfg.images_csv, cfg.image_root, tokenizer, cfg.context_length)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    model = ReportGenerator(vocab_size, cfg.model_dim, cfg.context_length, cfg.num_blocks, cfg.num_heads).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for step in range(cfg.train_steps):
        for images, captions in dataloader:
            images, captions = images.to(cfg.device), captions.to(cfg.device)
            optimizer.zero_grad()
            logits = model(images, captions[:,:-1])
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), captions[:,1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if step % 10 == 0:
            print(f"Step {step} | Loss {loss.item():.4f}")

    torch.save(model.state_dict(), "report_generator_final.pth")
    torch.save(tokenizer.stoi, "tokenizer_stoi.pth")
    torch.save(tokenizer.itos, "tokenizer_itos.pth")
    return model, tokenizer


# -----------------------------
# Generation
# -----------------------------
@torch.no_grad()
def generate_report(model, tokenizer, image_path, max_len=128):
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(cfg.device)
    tokens = [0] * max_len
    tokens = torch.tensor([tokens], dtype=torch.long, device=cfg.device)
    generated = []
    for i in range(max_len):
        logits = model(img, tokens)
        probs = F.softmax(logits[:,i,:], dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)
        tokens[0,i] = next_tok
        generated.append(next_tok.item())
    return tokenizer.decode(generated)

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    model, tokenizer = train()
    # Example generation
    example_image = "/Users/anishrajumapathy/Downloads/chest_xray/val/PNEUMONIA/rg.jpg"  # replace
    report = generate_report(model, tokenizer, example_image)
    print("Generated Report:\n", report)
