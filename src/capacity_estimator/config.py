import os
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Union
load_dotenv()
class Settings(BaseModel):
    gemini_api_key: Union[str, None] = os.getenv("GEMINI_API_KEY")
    gemini_model: str = os.getenv("GEMINI_MODEL","gemini-2.5-flash")
    yolo_model: str = os.getenv("YOLO_MODEL","yolov8n-seg.pt")
    yolo_conf: float = float(os.getenv("YOLO_CONF","0.25"))
    yolo_imgsz: Union[int, None] = int(os.getenv("YOLO_IMGSZ")) if os.getenv("YOLO_IMGSZ") else None
    outlines_dir: str = os.getenv("OUTLINES_DIR","outputs/outlines")
def get_settings() -> Settings:
    return Settings()
