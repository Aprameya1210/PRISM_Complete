import os, cv2
def imwrite_safely(p, img):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    cv2.imwrite(p, img)
