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
    width, height = image.size
    total_image_area = width * height

    # Run inference
    results = model(image)
    boxes = results[0].boxes
    names = model.names

    # Draw annotated image
    annotated_frame = results[0].plot()

    # Prepare outputs
    damage_info = []
    for box, cls_id in zip(boxes.xywh, boxes.cls):
        # box: (x_center, y_center, box_width, box_height)
        box_width = float(box[2])
        box_height = float(box[3])
        box_area = box_width * box_height
        normalized_area = box_area / total_image_area

        damage_info.append({
            "damage_type": names[int(cls_id)],
            "box_width": box_width,
            "box_height": box_height,
            "box_area": box_area,
            "normalized_area": normalized_area
        })

    # Convert annotated image to base64
    buffered = io.BytesIO()
    Image.fromarray(annotated_frame).save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return {
        "detections": damage_info,
        "image": img_str
    }
