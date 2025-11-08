import os, cv2
class OutlineDrawer:
    def draw_and_save(self, bgr, cnt, out_dir, base):
        os.makedirs(out_dir, exist_ok=True)
        out=bgr.copy()
        cv2.drawContours(out,[cnt],-1,(0,255,0),2)
        p=os.path.join(out_dir,f"{base}_outline.png")
        cv2.imwrite(p,out)
        return p
