import cv2, numpy as np
from ultralytics import YOLO
_yolo_cache={}
class YOLOMaskExtractor:
    def __init__(self, model_name="yolov8n-seg.pt", conf=0.25, imgsz=None):
        self.model_name=model_name
        self.conf=conf
        self.imgsz=imgsz
        if model_name not in _yolo_cache:
            _yolo_cache[model_name]=YOLO(model_name)
        self.model=_yolo_cache[model_name]
    def extract(self, bgr):
        h,w=bgr.shape[:2]
        imgsz=self.imgsz or max(640,min(h,w))
        res=self.model.predict(bgr, imgsz=imgsz, conf=self.conf, iou=0.5, verbose=False)[0]
        if res.masks is None or len(res.masks.data)==0:
            raise RuntimeError("YOLO-Seg found no instances")
        raw=res.masks.data.cpu().numpy()
        masks=[cv2.resize((m>0.5).astype(np.uint8)*255,(w,h),interpolation=cv2.INTER_NEAREST) for m in raw]
        names=res.names if isinstance(res.names,dict) else {i:n for i,n in enumerate(res.names)}
        cls_ids=res.boxes.cls.cpu().numpy().astype(int) if res.boxes.cls is not None else []
        pick=None
        for i,cid in enumerate(cls_ids):
            if str(names.get(cid,"")).lower()=="bottle":
                pick=i
                break
        if pick is None:
            areas=[int(m.sum()) for m in masks]
            pick=int(np.argmax(areas))
        mask=masks[pick]
        mask=cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8), iterations=2)
        cnts,_=cv2.findContours((mask>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            raise RuntimeError("Empty mask after postprocess")
        cnt=max(cnts, key=cv2.contourArea)
        return (mask>0).astype(np.uint8), cnt