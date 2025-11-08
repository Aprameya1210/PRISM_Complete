import cv2, numpy as np
class ContourMaskExtractor:
    def extract(self, gray):
        g=cv2.equalizeHist(gray)
        thr=cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,2)
        thr=cv2.medianBlur(thr,5)
        thr=cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8), iterations=2)
        cnts,_=cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            raise RuntimeError("No contours found")
        cnt=max(cnts, key=cv2.contourArea)
        mask=np.zeros_like(thr)
        cv2.drawContours(mask,[cnt],-1,255,thickness=cv2.FILLED)
        return (mask>0).astype(np.uint8), cnt