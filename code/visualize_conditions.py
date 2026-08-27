from __future__ import annotations
import argparse
from pathlib import Path
from PIL import Image,ImageDraw
from bbox_data import BBoxVehicleDataset,read_annotation
from rgb2sar.data import rgba_to_rgb
from swap_infer import generate_swap

def main():
 p=argparse.ArgumentParser();p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--rgb-root",type=Path,required=True);p.add_argument("--sar-root",type=Path,required=True);p.add_argument("--source-class",required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--device",default="cuda:0");a=p.parse_args();ds=BBoxVehicleDataset(a.rgb_root,a.sar_root,epoch_size=1)
 wanted=[(0,15),(60,30),(120,45),(180,60),(240,30),(300,45)];chosen=[]
 for target_az,target_dep in wanted:
  best=min(ds.records,key=lambda r:min(abs(r[3]["azimuth"]-target_az),360-abs(r[3]["azimuth"]-target_az))+20*abs(r[3]["depression"]-target_dep));chosen.append(best)
 size=128;canvas=Image.new("RGB",(size*4,32+size*len(chosen)),"white");draw=ImageDraw.Draw(canvas)
 for i,t in enumerate(("RGB source A","Host SAR","Generated A ROI","Replaced SAR")):draw.text((i*size+3,5),t,fill="black")
 source=a.rgb_root/a.source_class;temp=a.output.parent/"_condition_tmp.png"
 for row,(host,_,bbox,meta) in enumerate(chosen):
  roi,result,box,attn=generate_swap(a.checkpoint,source,host,host.with_suffix(".xml"),temp,a.device);rgb=Image.open(next(source.glob("*.png")));rgb.thumbnail((size,size),Image.Resampling.LANCZOS);rgb=rgba_to_rgb(rgb);tile=Image.new("RGB",(size,size),(127,127,127));tile.paste(rgb,((size-rgb.width)//2,(size-rgb.height)//2));marked=Image.open(host).convert("RGB");ImageDraw.Draw(marked).rectangle(box,outline="red",width=2);label=f"az={meta['azimuth']} dep={meta['depression']}";ImageDraw.Draw(marked).rectangle((0,0,92,13),fill="black");ImageDraw.Draw(marked).text((2,1),label,fill="white")
  y=32+row*size;canvas.paste(tile,(0,y));canvas.paste(marked,(size,y));canvas.paste(roi.resize((size,size)).convert("RGB"),(size*2,y));canvas.paste(result.convert("RGB"),(size*3,y))
 a.output.parent.mkdir(parents=True,exist_ok=True);canvas.save(a.output);temp.unlink(missing_ok=True);print(a.output)
if __name__=="__main__":main()
