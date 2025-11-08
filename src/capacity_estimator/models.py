from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel
class ScaleEstimate(BaseModel):
    can_estimate: bool
    mm_per_px: Optional[float]=None
    method: Optional[str]=None
    confidence: Optional[float]=None
    rationale: Optional[str]=None
    assumptions: Optional[List[str]]=None
class Result(BaseModel):
    volume: float
    height: float
    units: Dict[str,str]
    notes: str
    rotation_applied_deg: float
    crop: List[int]
    outline_path: Union[str, None] = None
    scale_mm_per_px: Union[float, None] = None
    gemini: Union[Dict[str,Any], None] = None
