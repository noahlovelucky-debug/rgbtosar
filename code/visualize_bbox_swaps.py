from __future__ import annotations
import argparse,random
from pathlib import Path
from PIL import Image,ImageDraw
from bbox_data import BBoxVehicleDataset
from rgb2sar.data import rgba_to_rgb
from swap_infer import generate_swap

def main():
 p=argparse.ArgumentParser();p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--rgb-root",type=Path,required=True);p.add_argument("--sar-root",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--samples",type=int,default=6);p.add_argument("--device",default="cuda:0");a=p.parse_args()
 ds=BBoxVehicleDataset(a.rgb_root,a.sar_root,epoch_size=1); rng=random.Random(7); records=rng.sample(ds.records,a.samples); size=128;canvas=Image.new("RGB",(size*4,28+size*a.samples),"white");d=ImageDraw.Draw(canvas)
 for x,label in enumerate(("Source RGB A","Host SAR B","Generated A ROI","A inserted into B")):d.text((x*size+3,7),label,fill="black")
 temp=a.output.parent/"_swap_temp.png"
 for row,(host,host_class,bbox,meta) in enumerate(records):
  source=rng.choice([c for c in ds.classes if c!=host_class]);source_dir=a.rgb_root/source;roi,result,box,attn=generate_swap(a.checkpoint,source_dir,host,host.with_suffix(".xml"),temp,a.device)
  rgb_path=next(source_dir.glob("*.png"));rgb=Image.open(rgb_path);rgb.thumbnail((size,size),Image.Resampling.LANCZOS);rgb=rgba_to_rgb(rgb);tile=Image.new("RGB",(size,size),(127,127,127));tile.paste(rgb,((size-rgb.width)//2,(size-rgb.height)//2));hostim=Image.open(host).convert("RGB");marked=hostim.copy();ImageDraw.Draw(marked).rectangle(box,outline="red",width=2)
  roi_tile=roi.resize((size,size)).convert("RGB");canvas.paste(tile,(0,28+row*size));canvas.paste(marked,(size,28+row*size));canvas.paste(roi_tile,(size*2,28+row*size));canvas.paste(result.convert("RGB"),(size*3,28+row*size))
 a.output.parent.mkdir(parents=True,exist_ok=True);canvas.save(a.output);temp.unlink(missing_ok=True);print(a.output)
if __name__=="__main__":main()
