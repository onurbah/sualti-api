from flask import Flask, request, jsonify
from ultralytics import YOLO
import pandas as pd
import os

app = Flask(__name__)

# Model ve bilgi dosyasını yükle
model = YOLO("best_yolov8.pt")
df = pd.read_csv("species_info.csv")

@app.route("/")
def home():
    return "🐠 Flask API ÇALIŞIYOR!"

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files['image']
    image_path = "temp.jpg"
    image.save(image_path)

    results = model(image_path)
    names = model.names
    boxes = results[0].boxes

    if len(boxes) == 0:
        return jsonify({"label": "none", "info": "No object detected"}), 200

    class_id = int(boxes[0].cls[0].item())
    label = names[class_id]

    info = df[df["label"] == label.lower()]
    if info.empty:
        return jsonify({"label": label, "info": "No info found"}), 200

    return jsonify({
        "label": label,
        "info": info.iloc[0].to_dict()
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)