import cv2, numpy as np
class ArucoScaleEstimator:
    def estimate_mm_per_px(self, bgr, marker_mm, dict_name=None):
        if not hasattr(cv2, "aruco"):
            raise RuntimeError("opencv-contrib missing for ArUco")
        if dict_name is None:
            dict_name=cv2.aruco.DICT_5X5_100
        gray=cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        ad=cv2.aruco.getPredefinedDictionary(dict_name)
        corners, ids, _=cv2.aruco.detectMarkers(gray, ad)
        if ids is None or len(corners)==0:
            raise RuntimeError("No ArUco marker detected")
        per=[cv2.arcLength(c.reshape(-1,2), True) for c in corners]
        idx=int(np.argmax(per))
        c=corners[idx].reshape(4,2)
        side=np.mean([np.linalg.norm(c[1]-c[0]), np.linalg.norm(c[2]-c[1]), np.linalg.norm(c[3]-c[2]), np.linalg.norm(c[0]-c[3])])
        return marker_mm/side
