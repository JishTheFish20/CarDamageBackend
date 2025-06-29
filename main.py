from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io
import base64

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
# Load model
model = YOLO("best.pt")

@app.post("/predict")
async def predict_damage(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Run inference and draw results on image
    results = model(image)
    annotated_frame = results[0].plot()  # draws bounding boxes
    names = model.names
    classes = results[0].boxes.cls.tolist()
    damage_types = [names[int(cls_id)] for cls_id in classes]

    # Convert annotated image to base64 string
    buffered = io.BytesIO()
    Image.fromarray(annotated_frame).save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return {
        "damage_type": damage_types,
        "image": img_str  # base64 encoded PNG
    }
