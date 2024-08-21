from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

# Serve static files from the 'static' directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Update the origins list to include your deployed front-end URL
origins = [
    "http://localhost",
    "http://localhost:3000",
    "https://potato-disease-classification-keu1.onrender.com"  # Ensure this is correct
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure the model path is correct
MODEL = tf.keras.models.load_model("models/model_v1.keras")

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert('RGB')
    image = image.resize((224, 224))  # Adjust size if necessary
    return np.array(image)

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

@app.get("/", response_class=HTMLResponse)
async def serve_html():
    with open("static/index.html", "r") as f:
        return HTMLResponse(f.read())

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)  # Update host to '0.0.0.0' to allow external access
