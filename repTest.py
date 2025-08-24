import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from transformers import GPT2Tokenizer

from report import CNNFeatureExtractor, SimpleReportGenerator  # replace with your filename

# ----------------------------
# Device setup
# ----------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# ----------------------------
# Tokenizer
# ----------------------------
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# ----------------------------
# Image transform (same as training)
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ----------------------------
# Load trained model
# ----------------------------
checkpoint_path = "/Users/anishrajumapathy/transformer/report_generator_epoch_5.pth"   # <-- change if needed
ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)


# Initialize CNN extractor (ResNet fallback since cnn_model_path=None)
cnn_extractor = CNNFeatureExtractor().to(device)
cnn_extractor.eval()

# Figure out CNN feature size
with torch.no_grad():
    dummy = torch.randn(1, 3, 224, 224).to(device)
    cnn_dim = cnn_extractor(dummy).size(1)

# Initialize report generator
model = SimpleReportGenerator(
    cnn_feature_dim=cnn_dim,
    vocab_size=tokenizer.vocab_size
).to(device)

model.load_state_dict(ckpt["model_state_dict"])
model.eval()

print(f"Loaded checkpoint from {checkpoint_path}")

# ----------------------------
# Load your test image manually
# ----------------------------
image_path = "/Users/anishrajumapathy/Downloads/chest_xray/val/PNEUMONIA/person1951_bacteria_4882.jpeg"   # image path 

img = Image.open(image_path).convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(device)

# ----------------------------
# Generate report
# ----------------------------
with torch.no_grad():
    cnn_features = cnn_extractor(img_tensor)
    generated_ids = model(cnn_features, generate_mode=True, max_new_tokens=120)
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# ----------------------------
# Show result
# ----------------------------
# De-normalize image for display
img_display = img_tensor[0].permute(1, 2, 0).cpu().numpy()
img_display = img_display * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
img_display = img_display.clip(0, 1)

plt.imshow(img_display)
plt.axis("off")
plt.title("Input Chest X-ray")
plt.show()

print("\n--- Generated Report ---\n")
print(generated_text)
