import base64
import io

from flask import Flask, jsonify, request, send_from_directory
from character_cnn.inference import ModelInference
from PIL import Image, ImageOps

model = ModelInference()

app = Flask(__name__, static_folder="frontend", static_url_path="")


@app.get("/")
def index():
    return send_from_directory("frontend", "index.html")


@app.post("/predict")
def predict():
    global model
    data_url = request.get_json()["image"]
    b64 = data_url.split(",", 1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("L")
    img = ImageOps.invert(img).resize((28, 28), Image.LANCZOS)
    img = img.transpose(Image.TRANSPOSE)
    label,confidence=model.inference(img)
    return jsonify(label=label, confidence=confidence)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
