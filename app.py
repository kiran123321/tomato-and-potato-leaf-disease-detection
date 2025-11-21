from flask import Flask, render_template, request, redirect, url_for
import onnxruntime as ort
from PIL import Image
import numpy as np
import os

app = Flask(__name__)
app.secret_key = "mysecretkey"

# -----------------------------
#  LOAD ONNX MODEL & LABELS
# -----------------------------
model_path = "model/model.onnx"
labels_path = "model/labels.txt"

session = ort.InferenceSession(model_path)

with open(labels_path, "r") as f:
    LABELS = [line.strip() for line in f.readlines()]

# -----------------------------
#  LOGIN PAGE
# -----------------------------
USERNAME = "admin"
PASSWORD = "admin"

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = request.form["username"]
        pwd = request.form["password"]

        if user == USERNAME and pwd == PASSWORD:
            return redirect(url_for("home"))
        else:
            return render_template("login.html", error="Invalid Credentials")

    return render_template("login.html")

# -----------------------------
#  HOME / UPLOAD PAGE
# -----------------------------
@app.route("/home")
def home():
    return render_template("home.html")

# -----------------------------
#  PREDICTION FUNCTION
# -----------------------------
def preprocess(img_path):
    img = Image.open(img_path).convert("RGB").resize((224,224))

    img = np.array(img).astype(np.float32) / 255.0

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std

    img = np.transpose(img, (2, 0, 1))   # HWC → CHW
    img = np.expand_dims(img, axis=0)    # Add batch dimension

    return img












# -----------------------------
#  RECOMMENDED PESTICIDES
# -----------------------------
PESTICIDES = {
    "Tomato___Early_blight": "Mancozeb 75% WP, Chlorothalonil, Copper oxychloride",
    "Tomato___Late_blight": "Metalaxyl + Mancozeb, Ridomil Gold, Propamocarb",
    "Tomato___Leaf_Mold": "Copper hydroxide, Chlorothalonil, Iprodione",
    "Tomato___Septoria_leaf_spot": "Chlorothalonil, Mancozeb, Thiophanate-methyl",
    "Tomato___Bacterial_spot": "Copper oxychloride, Bordeaux mixture, Streptocycline",
    "Tomato___Spider_mites": "Abamectin, Fenpyroximate, Spiromesifen",
    "Tomato___Target_Spot": "Azoxystrobin, Difenoconazole, Mancozeb",
    "Tomato___Yellow_Leaf_Curl_Virus": "Imidacloprid, Thiamethoxam",
    "Tomato___Mosaic_Virus": "No cure — remove infected plants; control aphids",
    "Tomato___healthy": "No pesticides needed",

    "Pepper__bell___Bacterial_spot": "Copper hydroxide + Mancozeb, Streptocycline",
    "Pepper__bell___healthy": "No pesticides needed",

    "Potato___Early_blight": "Mancozeb, Chlorothalonil, Copper oxychloride",
    "Potato___Late_blight": "Propamocarb, Cymoxanil + Mancozeb, Metalaxyl",
    "Potato___healthy": "No pesticides needed",
}




























@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return "No file uploaded!"

    file = request.files["image"]
    filepath = "static/uploaded.jpg"
    file.save(filepath)

    img = preprocess(filepath)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    pred = session.run([output_name], {input_name: img})[0]
    pred_idx = np.argmax(pred)

    predicted_label = LABELS[pred_idx]

    pesticide = PESTICIDES.get(predicted_label, "No data available")

    return render_template("result.html",
                       image="uploaded.jpg",
                       label=predicted_label,
                       pesticide=pesticide)


# -----------------------------
#  RUN APP
# -----------------------------
if __name__ == "__main__":
     port = int(os.environ.get("PORT", 5000))
     app.run(host="0.0.0.0", port=port)
