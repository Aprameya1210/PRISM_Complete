# from typing import Union
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware

# from routes import index
# from routes.api import detect

# app = FastAPI()

# # Configure CORS
# origins = ["*"]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# app.include_router(index.router)
# app.include_router(detect.router, prefix="/api")


import argparse
import sys
import json
import os
from PIL import Image

from server.yolo.yolo import YOLOModel

def generate_clustering_image(result):
    clustering_image_array = result.plot(boxes=False, labels=True, color_mode="class")
    clustering_image_array_rgb = clustering_image_array[..., ::-1] 
    clustering_image = Image.fromarray(clustering_image_array_rgb)
    return clustering_image

def run_detection(image_path: str, yolo_model: YOLOModel):
    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: Could not open image. {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing image: {image_path}...")
    
    detected_objects, results = yolo_model.predict(image)

    if results is None:
        print("Error: Error during object detection.", file=sys.stderr)
        sys.exit(1)

    output_dir = "outlines"
    
    os.makedirs(output_dir, exist_ok=True) 

    output_image_path = os.path.join(output_dir, "output_image.jpg")
    clustering_image_path = os.path.join(output_dir, "clustering_image.jpg")
    
    results[0].save(output_image_path)
    clustering_image = generate_clustering_image(results[0])
    clustering_image.save(clustering_image_path)

    print(f"Saved detection image to: {output_image_path}")
    print(f"Saved clustering image to: {clustering_image_path}")

    garbage_classes = {35.0}
    ignore_classes = {58.0, 31.0, 42.0, 70.0, 83.0, 25.0, 27.0, 22.0, 11.0, 8.0}
    plate_area = 0
    garbage_area = 0
    food_area = 0

    for obj in detected_objects:
        if obj["label"] == 58.0:
            plate_area += obj["area"]
        elif obj["label"] in garbage_classes:
            garbage_area += obj["area"]
        elif obj["label"] not in ignore_classes:
            food_area += obj["area"]

    if plate_area == 0:
        print("Error: No plate detected in the image.", file=sys.stderr)
        return {"error": "No plate detected in the image"}

    if plate_area > garbage_area:
        food_percentage = (food_area / (plate_area - garbage_area)) * 100
    else:
        food_percentage = 0

    food_percentage = min(max(food_percentage, 0), 100)
    
    return {
        "food_percentage": round(food_percentage, 2),
        "food_area": food_area,
        "garbage_area": garbage_area,
        "plate_area": plate_area,
        "detected_objects_count": len(detected_objects),
        "objects": detected_objects,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run food percentage detection on an image.")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input image file."
    )
    
    args = parser.parse_args()

    print("Loading YOLO model...")
    try:
        model = YOLOModel()
        if model.model is None:
            raise Exception("Model initialization failed.")
    except Exception as e:
        print(f"Fatal: Could not load YOLO model. {e}", file=sys.stderr)
        sys.exit(1)
    print("Model loaded successfully.")

    detection_results = run_detection(args.image, model)

    print("\n--- Detection Results ---")
    print(json.dumps(detection_results, indent=2))