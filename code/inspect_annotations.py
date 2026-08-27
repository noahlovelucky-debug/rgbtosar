from __future__ import annotations
import argparse
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image

def text(root: ET.Element, *paths: str) -> str | None:
    for path in paths:
        node = root.find(path)
        if node is not None and node.text: return node.text.strip()
    return None

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--sar-root", type=Path, required=True); p.add_argument("--limit", type=int, default=0); args=p.parse_args()
    widths=[]; heights=[]; centers=[]; samples=[]
    for i,path in enumerate(args.sar_root.rglob("*.xml")):
        if args.limit and i >= args.limit: break
        try: root=ET.parse(path).getroot()
        except ET.ParseError: continue
        vals=[text(root, f".//{k}") for k in ("xmin","ymin","xmax","ymax")]
        if any(v is None for v in vals): continue
        xmin,ymin,xmax,ymax=map(float,vals); widths.append(xmax-xmin); heights.append(ymax-ymin); centers.append(((xmin+xmax)/2,(ymin+ymax)/2))
        if len(samples)<5:
            tif=path.with_suffix(".tif"); samples.append((path.name, Image.open(tif).size if tif.exists() else None, vals,
                text(root,".//target_azimuth_angle"), text(root,".//type",".//subclass")))
    print("valid",len(widths)); print("bbox_width min/median/max",min(widths),statistics.median(widths),max(widths)); print("bbox_height min/median/max",min(heights),statistics.median(heights),max(heights));
    print("center_x min/median/max",min(x for x,y in centers),statistics.median(x for x,y in centers),max(x for x,y in centers)); print("center_y min/median/max",min(y for x,y in centers),statistics.median(y for x,y in centers),max(y for x,y in centers)); print("samples",samples)
if __name__=="__main__": main()
