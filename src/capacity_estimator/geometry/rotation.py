import cv2, math, numpy as np
class RotationAligner:
    def rotate_to_vertical(self, img, mask, cnt):
        pts=cnt.reshape(-1,2).astype(np.float32)
        _, eig=cv2.PCACompute(pts, mean=None)
        v=eig[0]
        ang=math.degrees(math.atan2(v[1], v[0]))
        rot=90-ang
        h,w=img.shape[:2]
        M=cv2.getRotationMatrix2D((w/2,h/2), rot, 1.0)
        ir=cv2.warpAffine(img, M, (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        mr=cv2.warpAffine(mask, M, (w,h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        return ir, mr, rot
