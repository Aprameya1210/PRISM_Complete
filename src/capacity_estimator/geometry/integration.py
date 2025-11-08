import cv2, math, numpy as np
class VolumeIntegrator:
    def integrate_px(self, mask):
        if mask.ndim==3:
            mask=cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        m=(mask>0).astype(np.uint8)
        rows=np.where(m.any(axis=1))[0]
        if rows.size<10:
            raise RuntimeError("Object too small")
        top,bot=rows[0],rows[-1]
        vol=0.0
        for y in range(top,bot+1):
            cols=np.where(m[y]>0)[0]
            if cols.size<2:
                continue
            left,right=cols[0],cols[-1]
            w=right-left+1
            r=w/2.0
            vol+=math.pi*(r**2)
        h=(bot-top+1)*1.0
        return vol,h
    def integrate_mm(self, mask, mm_per_px, wall_mm=0.0):
        if mask.ndim==3:
            mask=cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        m=(mask>0).astype(np.uint8)
        rows=np.where(m.any(axis=1))[0]
        if rows.size<10:
            raise RuntimeError("Object too small")
        top,bot=rows[0],rows[-1]
        vol=0.0
        for y in range(top,bot+1):
            cols=np.where(m[y]>0)[0]
            if cols.size<2:
                continue
            left,right=cols[0],cols[-1]
            w=right-left+1
            r=(w/2.0)*mm_per_px-wall_mm
            if r<0:
                r=0.0
            vol+=math.pi*(r**2)*mm_per_px
        h=(bot-top+1)*mm_per_px
        return vol,h
