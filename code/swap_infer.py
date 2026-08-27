from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from PIL import Image,ImageFilter,ImageDraw
import torch
from bbox_data import load_multiview,read_annotation,metadata_vector
from bbox_models import MultiViewEncoder,ROIGenerator

def load_model(checkpoint_path:Path,device:str):
 ck=torch.load(checkpoint_path,map_location=device,weights_only=False);enc=MultiViewEncoder().to(device);gen=ROIGenerator().to(device);enc.load_state_dict(ck["encoder"]);gen.load_state_dict(ck["generator"]);enc.eval();gen.eval();return enc,gen,ck

def generate_swap(checkpoint:Path,source_rgb_dir:Path,host_sar:Path,host_xml:Path,output:Path,device="cuda:0"):
 enc,gen,ck=load_model(checkpoint,device); bbox,meta=read_annotation(host_xml); views,mask=load_multiview(source_rgb_dir,64); mv=metadata_vector(meta,bbox)
 with torch.inference_mode(): identity,attn=enc(views[None].to(device),mask[None].to(device),mv[None].to(device)); roi=gen(identity,mv[None].to(device))[0,0].cpu()
 roi_img=Image.fromarray(((roi.clamp(-1,1).numpy()+1)*127.5).astype(np.uint8),"L"); base=Image.open(host_sar).convert("L"); xmin,ymin,xmax,ymax=bbox; w,h=xmax-xmin,ymax-ymin
 generated=roi_img.resize((w,h),Image.Resampling.BILINEAR)
 # Feather only the boundary: the generated ROI replaces the old target in the center.
 alpha=Image.new("L",(w,h),255); border=max(2,min(w,h)//10); d=ImageDraw.Draw(alpha); d.rectangle((0,0,w-1,h-1),outline=0,width=border);alpha=alpha.filter(ImageFilter.GaussianBlur(border/2))
 result=base.copy(); old=base.crop(bbox); blended=Image.composite(generated,old,alpha);result.paste(blended,(xmin,ymin));output.parent.mkdir(parents=True,exist_ok=True);result.save(output)
 return roi_img,result,bbox,attn[0].cpu().tolist()

def main():
 p=argparse.ArgumentParser();p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--source-rgb-dir",type=Path,required=True);p.add_argument("--host-sar",type=Path,required=True);p.add_argument("--host-xml",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--device",default="cuda:0");a=p.parse_args();roi,result,bbox,attn=generate_swap(a.checkpoint,a.source_rgb_dir,a.host_sar,a.host_xml,a.output,a.device);roi.save(a.output.with_name(a.output.stem+"_roi.png"));print("bbox",bbox,"view_attention",[round(x,3) for x in attn],"output",a.output)
if __name__=="__main__":main()

