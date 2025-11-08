import os, json, cv2, numpy as np
from .config import get_settings
from .models import Result
from .image_io import imwrite_safely
from .masks import YOLOMaskExtractor, ContourMaskExtractor
from .geometry import RotationAligner, VolumeIntegrator
from .scale import ArucoScaleEstimator, GeminiScaleEstimator
from .viz import OutlineDrawer
from typing import Union
class CapacityPipeline:
    def __init__(self, yolo_model:Union[str,None]=None, yolo_conf:Union[float,None]=None, yolo_imgsz:Union[int,None]=None, outlines_dir:Union[str,None]=None, gemini_api_key:Union[str,None]=None, gemini_model:Union[str,None]=None):
        s=get_settings()
        self.outlines_dir=outlines_dir or s.outlines_dir
        self.yolo=YOLOMaskExtractor(model_name=yolo_model or s.yolo_model, conf=yolo_conf if yolo_conf is not None else s.yolo_conf, imgsz=yolo_imgsz if yolo_imgsz is not None else s.yolo_imgsz)
        self.contour=ContourMaskExtractor()
        self.rot=RotationAligner()
        self.intg=VolumeIntegrator()
        self.aruco=ArucoScaleEstimator()
        self.gemini=GeminiScaleEstimator(api_key=gemini_api_key if gemini_api_key is not None else s.gemini_api_key, model=gemini_model or s.gemini_model)
        self.drawer=OutlineDrawer()
    def process(self, image_path:str, mm_per_px:Union[float,None], aruco_mm:Union[float,None], wall_mm:float, debug_dir:Union[str,None], crop_margin:int, use_gemini:bool):
        bgr=cv2.imread(image_path)
        if bgr is None:
            raise FileNotFoundError(image_path)
        base=os.path.splitext(os.path.basename(image_path))[0]
        gray=cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray_blur=cv2.GaussianBlur(gray,(5,5),0)
        try:
            mask_bin, cnt=self.yolo.extract(bgr)
        except Exception:
            mask_bin, cnt=self.contour.extract(gray_blur)
        outline_path=self.drawer.draw_and_save(bgr, cnt, self.outlines_dir, base)
        _, mask_r, rot=self.rot.rotate_to_vertical(bgr, (mask_bin*255).astype(np.uint8), cnt)
        ys,xs=np.where(mask_r>0)
        if ys.size==0 or xs.size==0:
            raise RuntimeError("Mask vanished after rotation")
        y1,y2=max(ys.min()-crop_margin,0), min(ys.max()+crop_margin, mask_r.shape[0]-1)
        x1,x2=max(xs.min()-crop_margin,0), min(xs.max()+crop_margin, mask_r.shape[1]-1)
        mask_r=mask_r[y1:y2+1, x1:x2+1]
        if debug_dir:
            imwrite_safely(os.path.join(debug_dir,"01_gray.jpg"), gray)
            imwrite_safely(os.path.join(debug_dir,"02_mask_raw.png"), (mask_bin*255).astype(np.uint8))
            imwrite_safely(os.path.join(debug_dir,"03_mask_rotated.png"), mask_r)
        gemini_info=None
        if mm_per_px is None and aruco_mm is not None:
            try:
                mm_per_px=self.aruco.estimate_mm_per_px(bgr, aruco_mm)
            except Exception:
                pass
        if mm_per_px is None and use_gemini:
            vpx,hpx=self.intg.integrate_px(mask_r)
            meta={"volume": float(vpx), "height": float(hpx), "units":{"volume":"px^3","height":"px"}, "notes":"No scale provided; pixel units.", "rotation_applied_deg": float(rot), "crop":[int(y1),int(y2),int(x1),int(x2)]}
            gemini_info=self.gemini.estimate_mm_per_px(image_path, meta)
            if isinstance(gemini_info,dict) and gemini_info.get("mm_per_px"):
                try:
                    mm_per_px=float(gemini_info["mm_per_px"])
                except:
                    pass
        if mm_per_px is None:
            vpx,hpx=self.intg.integrate_px(mask_r)
            return Result(volume=float(vpx), height=float(hpx), units={"volume":"px^3","height":"px"}, notes="No scale provided; reporting in pixel units.", rotation_applied_deg=float(rot), crop=[int(y1),int(y2),int(x1),int(x2)], outline_path=outline_path, gemini=gemini_info).model_dump()
        vmm3,hmm=self.intg.integrate_mm(mask_r, mm_per_px, wall_mm)
        vml=vmm3/1000.0
        return Result(volume=float(vml), height=float(hmm), units={"volume":"mL","height":"mm"}, notes=("Inner capacity (wall subtracted)." if wall_mm>0 else "Outer volume (no wall subtraction)."), rotation_applied_deg=float(rot), crop=[int(y1),int(y2),int(x1),int(x2)], outline_path=outline_path, scale_mm_per_px=float(mm_per_px), gemini=gemini_info).model_dump()
