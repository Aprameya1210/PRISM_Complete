import argparse, os, json
from .pipeline import CapacityPipeline
def main():
    ap=argparse.ArgumentParser(description="Gemini scale -> YOLO mask -> single-view capacity")
    ap.add_argument("--image", required=True)
    ap.add_argument("--mm-per-px", type=float, default=None)
    ap.add_argument("--aruco-mm", type=float, default=None)
    ap.add_argument("--wall-mm", type=float, default=0.0)
    ap.add_argument("--debug", type=str, default=None)
    ap.add_argument("--crop-margin", type=int, default=20)
    ap.add_argument("--use-gemini", action="store_true")
    ap.add_argument("--gemini-model", type=str, default=None)
    ap.add_argument("--outlines-dir", type=str, default=None)
    ap.add_argument("--yolo-model", type=str, default=None)
    ap.add_argument("--yolo-conf", type=float, default=None)
    ap.add_argument("--yolo-imgsz", type=int, default=None)
    args=ap.parse_args()
    pipe=CapacityPipeline(yolo_model=args.yolo_model, yolo_conf=args.yolo_conf, yolo_imgsz=args.yolo_imgsz, outlines_dir=args.outlines_dir, gemini_model=args.gemini_model)
    res=pipe.process(args.image, args.mm_per_px, args.aruco_mm, args.wall_mm, args.debug, args.crop_margin, args.use_gemini)
    vu=res["units"]["volume"]; hu=res["units"]["height"]
    if vu=="px^3":
        print(f"Estimated volume: {res['volume']:.0f} {vu}")
        print(f"Estimated height: {res['height']:.1f} {hu}")
        if "gemini" in res and isinstance(res["gemini"],dict) and res["gemini"] and res["gemini"].get("mm_per_px"):
            print(f"Gemini scale estimate: {float(res['gemini']['mm_per_px']):.6f} mm/px")
        print(res["notes"])
    else:
        print(f"Estimated volume: {res['volume']:.2f} {vu}")
        print(f"Estimated height: {res['height']:.1f} {hu}")
        if "scale_mm_per_px" in res and res["scale_mm_per_px"] is not None:
            print(f"Scale: {res['scale_mm_per_px']:.6f} mm/px")
        print(res["notes"])
    try:
        out_json=os.path.splitext(args.image)[0]+"_capacity.json"
        with open(out_json,"w") as f:
            json.dump(res,f,indent=2)
        print(f"Saved: {out_json}")
        if "outline_path" in res and res["outline_path"]:
            print(f"Saved outline: {res['outline_path']}")
    except Exception:
        pass
if __name__=="__main__":
    main()
