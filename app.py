from flask_cors import CORS
import torch
import torchvision
from torchvision import transforms
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from huggingface_hub import hf_hub_download
import os

class_name = []
with open("classes.txt") as f:
    for line in f:
        class_name.append(line.strip().split()[1])

model_path = hf_hub_download(repo_id="vvkon/bird-species-classifier", filename="model.pth")
model = torchvision.models.resnet18(weights=None)
model.fc = torch.nn.Linear(512, 200)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

data_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225],
    ),
])

app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    img = Image.open(file).convert("RGB")
    img = data_transforms(img)
    img = img.unsqueeze(0)
    with torch.no_grad():
        output = model(img)
        predicted_idx = torch.argmax(output, dim=1).item()
        confidence = torch.softmax(output, dim=1)[0][predicted_idx].item()
        species = class_name[predicted_idx]
    return jsonify({"species": species, "confidence": confidence})

if __name__ == "__main__":
    app.run(debug=True)