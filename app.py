from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

import tensorflow as tf
import numpy as np
from PIL import Image
import io

# IMPORTANT: Use the SAME preprocessing as training
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

# --------------------------------------------------
# FastAPI App
# --------------------------------------------------
app = FastAPI(title="Osteoarthritis Severity Detection API")

# Serve frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def home():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

# --------------------------------------------------
# Load Model
# --------------------------------------------------
model = tf.keras.models.load_model("model/model.keras")

print("MODEL OUTPUT SHAPE:", model.output_shape)  # Debug info

# --------------------------------------------------
# KL Grade Labels
# --------------------------------------------------
CLASS_NAMES = {
    0: "No Osteoarthritis",
    1: "Doubtful OA",
    2: "Mild OA",
    3: "Moderate OA",
    4: "Severe OA"
}

# --------------------------------------------------
# Image Preprocessing (MATCHES TRAINING)
# --------------------------------------------------
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((224, 224))   # MUST match training
    img_array = np.array(image)
    img_array = preprocess_input(img_array)  # NO /255 here
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --------------------------------------------------
# Prediction Endpoint
# --------------------------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    input_tensor = preprocess_image(image)
    predictions = model.predict(input_tensor)[0]

    predicted_class = int(np.argmax(predictions))
    confidence = float(predictions[predicted_class])

    # Return all probabilities (very useful)
    probabilities = {
        CLASS_NAMES[i]: round(float(predictions[i]), 4)
        for i in range(len(predictions))
    }

    return {
        "kl_grade": predicted_class,
        "severity": CLASS_NAMES[predicted_class],
        "confidence": round(confidence, 3),
        "all_probabilities": probabilities
    }
