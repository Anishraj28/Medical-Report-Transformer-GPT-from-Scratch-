import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from transformers import GPT2Tokenizer

class IndianaXrayDataset(Dataset):
    """
    Dataset for Indiana X-ray images and reports
    """
    def __init__(self, reports_csv, images_dir, tokenizer, transform=None, max_length=256):
        """
        Args:
            reports_csv: Path to indiana_reports.csv
            images_dir: Path to images_normalized folder
            tokenizer: Tokenizer for reports
            transform: Image transforms
            max_length: Maximum report length
        """
        # Load reports CSV
        self.reports_df = pd.read_csv(reports_csv)
        self.images_dir = images_dir
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length
        
        # Clean and prepare data
        self.prepare_data()
    
    def prepare_data(self):
        """Clean and prepare the data"""
        # Remove rows with empty findings/impressions
        self.reports_df = self.reports_df.dropna(subset=['findings', 'impression'])
        
        # Combine findings and impression into one report
        self.reports_df['full_report'] = (
            "FINDINGS: " + self.reports_df['findings'].astype(str) + 
            " IMPRESSION: " + self.reports_df['impression'].astype(str)
        )
        
        # Filter to only include images that exist
        valid_indices = []
        for idx, row in self.reports_df.iterrows():
            # Check if corresponding image exists
            image_files = self.get_image_files(row['uid'])
            if image_files:  # If at least one image exists
                valid_indices.append(idx)
        
        self.reports_df = self.reports_df.loc[valid_indices].reset_index(drop=True)
        print(f"Dataset size after filtering: {len(self.reports_df)}")
    
    def get_image_files(self, uid):
        """Get image files for a given UID"""
        # Images are named like: UID_IM-XXXX-XXXX.dcm.png
        possible_files = []
        for filename in os.listdir(self.images_dir):
            if filename.startswith(str(uid)) and filename.endswith('.png'):
                possible_files.append(filename)
        return possible_files
    
    def __len__(self):
        return len(self.reports_df)
    
    def __getitem__(self, idx):
        row = self.reports_df.iloc[idx]
        uid = row['uid']
        report = row['full_report']
        
        # Get image files
        image_files = self.get_image_files(uid)
        
        # Load first available image
        image_path = os.path.join(self.images_dir, image_files[0])
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Tokenize report
        encoding = self.tokenizer(
            report,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'image': image,
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'report_text': report
        }


class SimpleReportGenerator(nn.Module):
    """
    Simple model: CNN features + Regular Transformer for report generation
    This is NOT a Vision Transformer - just your regular transformer!
    """
    def __init__(self, cnn_feature_dim=512, vocab_size=50257, max_length=256, 
                 d_model=512, nhead=8, num_layers=6):
        super().__init__()
        
        self.d_model = d_model
        self.max_length = max_length
        
        # Project CNN features to transformer dimension
        self.cnn_projection = nn.Linear(cnn_feature_dim, d_model)
        
        # Regular transformer components (like your GPT)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_length, d_model)
        
        # Transformer decoder layers (just like your GPT!)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Output projection to vocabulary
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Layer norm
        self.ln_f = nn.LayerNorm(d_model)
    
    def forward(self, cnn_features, input_ids=None, generate_mode=False, max_new_tokens=100):
        """
        Forward pass
        Args:
            cnn_features: Features from CNN (batch_size, cnn_feature_dim)
            input_ids: Token IDs for training (batch_size, seq_len)
            generate_mode: Whether to generate text
        """
        batch_size = cnn_features.size(0)
        device = cnn_features.device
        
        # Project CNN features to transformer dimension
        visual_features = self.cnn_projection(cnn_features)  # (B, d_model)
        visual_features = visual_features.unsqueeze(1)  # (B, 1, d_model)
        
        if generate_mode:
            return self.generate_text(visual_features, max_new_tokens)
        
        # Training mode - teacher forcing
        seq_len = input_ids.size(1)
        
        # Token embeddings
        token_embeds = self.token_embedding(input_ids)  # (B, T, d_model)
        
        # Position embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        pos_embeds = self.position_embedding(positions)
        
        # Combine token and position embeddings
        text_embeds = token_embeds + pos_embeds  # (B, T, d_model)
        
        # Create causal mask for autoregressive generation
        tgt_mask = self.generate_square_subsequent_mask(seq_len).to(device)
        
        # Transformer decoder
        # Memory = visual features, tgt = text embeddings
        output = self.transformer_decoder(
            tgt=text_embeds,
            memory=visual_features,
            tgt_mask=tgt_mask
        )
        
        output = self.ln_f(output)
        logits = self.output_projection(output)
        
        return logits
    
    def generate_text(self, visual_features, max_new_tokens):
        """Generate text given visual features"""
        batch_size = visual_features.size(0)
        device = visual_features.device
        
        # Start with BOS token (assuming token ID 50256 is BOS for GPT-2)
        generated = torch.full((batch_size, 1), 50256, device=device, dtype=torch.long)
        
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                seq_len = generated.size(1)
                
                # Token and position embeddings
                token_embeds = self.token_embedding(generated)
                positions = torch.arange(seq_len, device=device).unsqueeze(0)
                pos_embeds = self.position_embedding(positions)
                text_embeds = token_embeds + pos_embeds
                
                # Causal mask
                tgt_mask = self.generate_square_subsequent_mask(seq_len).to(device)
                
                # Forward pass
                output = self.transformer_decoder(
                    tgt=text_embeds,
                    memory=visual_features,
                    tgt_mask=tgt_mask
                )
                
                output = self.ln_f(output)
                logits = self.output_projection(output)
                
                # Get last token probabilities
                last_token_logits = logits[:, -1, :]
                
                # Sample next token (you can add temperature, top-k sampling here)
                next_token = torch.multinomial(F.softmax(last_token_logits, dim=-1), 1)
                
                # Append to sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Check for EOS token (assuming 50256 is EOS)
                if next_token.item() == 50256:
                    break
        
        return generated
    
    def generate_square_subsequent_mask(self, sz):
        """Generate causal mask"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))


class CNNFeatureExtractor(nn.Module):
    """
    Extract features from your trained CNN
    """
    def __init__(self, trained_cnn_path=None):
        super().__init__()
        
        if trained_cnn_path and os.path.exists(trained_cnn_path):
            # Load your trained CNN
            print(f"Loading trained CNN from {trained_cnn_path}")
            checkpoint = torch.load(trained_cnn_path, map_location='cpu')
            
            # Create your CNN (you'll need to import your class here)
            # For now, using a placeholder - replace with your actual CNN class
            from your_cnn_file import MultiDiseaseConvolutionalNetwork
            self.cnn = MultiDiseaseConvolutionalNetwork(num_classes=15)
            self.cnn.load_state_dict(checkpoint['model_state_dict'])
            
            # Remove final classification layer to get features
            # Modify based on your CNN architecture
            self.feature_layers = nn.Sequential(*list(self.cnn.children())[:-1])
            
        else:
            # Use a pre-trained ResNet as backup
            print("Using pre-trained ResNet50 as feature extractor")
            import torchvision.models as models
            resnet = models.resnet50(pretrained=True)
            self.feature_layers = nn.Sequential(*list(resnet.children())[:-1])
        
        # Global average pooling to get fixed-size features
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Freeze CNN weights initially
        for param in self.feature_layers.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        features = self.feature_layers(x)
        
        # Handle different output shapes
        if len(features.shape) > 2:
            features = self.global_pool(features)
            features = features.flatten(1)
        
        return features


def train_report_generator(
    reports_csv_path,
    images_dir,
    cnn_model_path=None,
    num_epochs=5,
    batch_size=2,
    learning_rate=1e-4
):
    """
    Training function
    """
    # Device setup for Apple Silicon (M3 chip)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    print(f"Using device: {device}")
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 doesn't have pad token
    
    # Image transforms (same as your CNN training)
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    dataset = IndianaXrayDataset(
        reports_csv=reports_csv_path,
        images_dir=images_dir,
        tokenizer=tokenizer,
        transform=transform
    )
    
    # Split dataset (80% train, 20% val)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Initialize models
    cnn_extractor = CNNFeatureExtractor(cnn_model_path).to(device)
    
    # Determine CNN output dimension
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        cnn_output = cnn_extractor(dummy_input)
        cnn_feature_dim = cnn_output.size(1)
        print(f"CNN feature dimension: {cnn_feature_dim}")
    
    model = SimpleReportGenerator(
        cnn_feature_dim=cnn_feature_dim,
        vocab_size=tokenizer.vocab_size
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training loop
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            
            # Extract CNN features
            with torch.no_grad():  # Don't train CNN initially
                cnn_features = cnn_extractor(images)
            
            # Forward pass (teacher forcing)
            logits = model(cnn_features, input_ids[:, :-1])  # Exclude last token for input
            targets = input_ids[:, 1:]  # Exclude first token for target
            
            # Calculate loss
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=tokenizer.pad_token_id
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        scheduler.step()
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'tokenizer': tokenizer,
            'loss': avg_loss
        }, f'report_generator_epoch_{epoch+1}.pth')
    
    return model, tokenizer


# Example usage
if __name__ == "__main__":
    # Paths - update these to your actual paths
    reports_csv_path = "/Users/anishrajumapathy/Downloads/archive-4/indiana_reports.csv"
    images_dir = "/Users/anishrajumapathy/Downloads/archive-4/images/images_normalized"
    cnn_model_path = None  # Optional
    
    # Train the model
    model, tokenizer = train_report_generator(
        reports_csv_path=reports_csv_path,
        images_dir=images_dir,
        cnn_model_path=cnn_model_path,
        num_epochs=20,
        batch_size=4,  # Start small
        learning_rate=1e-4
    )
    
    print("Training completed!")