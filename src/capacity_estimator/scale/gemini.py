import json, mimetypes
from google import genai
from google.genai import types
from pydantic import BaseModel
from ..models import ScaleEstimate
from typing import Union
class GeminiScaleEstimator:
    def __init__(self, api_key: Union[str, None], model: str):
        self.api_key=api_key
        self.model=model
    def estimate_mm_per_px(self, image_path: str, meta: dict):
        if not self.api_key:
            return {"error":"No GEMINI_API_KEY configured"}
        try:
            client=genai.Client(api_key=self.api_key)
            mime=mimetypes.guess_type(image_path)[0] or "image/jpeg"
            with open(image_path,"rb") as f:
                data=f.read()
            part=types.Part.from_bytes(data=data, mime_type=mime)
            prompt=("You are a metrology assistant. Estimate a plausible pixel-to-metric scale for the container/bottle in the provided image.\n"
                    "Use the following metadata from a solid-of-revolution integration result:\n"
                    +json.dumps(meta,indent=2)+
                    "\nIf you can identify any reference of known size in the image (ruler, coin, A4/letter paper, cap standards, labels, ArUco), prefer that and output mm_per_px. "
                    "If no reliable reference is visible, provide your best estimate with assumptions; else set can_estimate=false. "
                    "Return only JSON with keys: can_estimate, mm_per_px, method, confidence, rationale, assumptions.")
            resp=client.models.generate_content(model=self.model, contents=[part, prompt], config={"response_mime_type":"application/json","response_schema":ScaleEstimate})
            try:
                parsed=resp.parsed
                d=parsed.model_dump()
            except Exception:
                d=json.loads(getattr(resp,"text","{}"))
            if isinstance(d,list) and d:
                d=d[0]
            if not isinstance(d,dict):
                d={}
            v=d.get("mm_per_px",None)
            if isinstance(v,str):
                try:
                    v=float(v)
                except:
                    v=None
            d["mm_per_px"]=v
            return d
        except Exception as e:
            return {"error":str(e)}
