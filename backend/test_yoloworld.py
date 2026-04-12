from ultralytics import YOLOWorld
import torch
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

print("Loading model...")
# Force CPU
model = YOLOWorld('yolov8s-world.pt')

classes = ["chair", "couch", "shelf", "lamp", "cushion", "rug", "drawer"]
model.set_classes(classes)

print("Running test prediction...")
import requests
from PIL import Image
from io import BytesIO

# Fetch a test image of a living room
url = "https://picsum.photos/seed/livingroom123/800/600"
response = requests.get(url)
img = Image.open(BytesIO(response.content))

results = model.predict(img, device='cpu')

for box in results[0].boxes:
    idx = int(box.cls[0].item())
    label = classes[idx]
    conf = float(box.conf[0].item())
    print(f"Detected: {label} ({conf:.2f})")

print("Done.")
