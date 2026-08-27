from __future__ import annotations
import argparse
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from bbox_data import BBoxVehicleDataset
from bbox_models import MultiViewEncoder,ROIGenerator
from sar_identity import SARIdentityEncoder
from align_rgb_sar import rgb_prototypes

def main():
 p=argparse.ArgumentParser();p.add_argument("--gan",type=Path,required=True);p.add_argument("--aligned",type=Path,required=True);p.add_argument("--rgb-root",type=Path,required=True);p.add_argument("--sar-root",type=Path,required=True);p.add_argument("--samples",type=int,default=5000);p.add_argument("--device",default="cuda:0");a=p.parse_args();dev=torch.device(a.device)
 gc=torch.load(a.gan,map_location=dev,weights_only=False);ac=torch.load(a.aligned,map_location=dev,weights_only=False);rgb=MultiViewEncoder().to(dev);gen=ROIGenerator().to(dev);sar=SARIdentityEncoder().to(dev);rp=torch.nn.Linear(256,128).to(dev);sp=torch.nn.Linear(256,128).to(dev)
 rgb.load_state_dict(ac["rgb_encoder"]);sar.load_state_dict(ac["sar_encoder"]);rp.load_state_dict(ac["rgb_projection"]);sp.load_state_dict(ac["sar_projection"]);gen.load_state_dict(gc["generator"])
 for m in (rgb,sar,rp,sp,gen):m.eval()
 prototypes=rgb_prototypes(rgb,rp,a.rgb_root,ac["classes"],dev);ds=BBoxVehicleDataset(a.rgb_root,a.sar_root,epoch_size=a.samples,pre_cropped=True);dl=DataLoader(ds,64,shuffle=True)
 fake1=fake5=real1=real5=total=0
 with torch.inference_mode():
  for b in dl:
   views,mask,meta,real,label=[b[k].to(dev) for k in ("views","view_mask","meta","roi","class_id")];identity,_=rgb(views,mask,meta);fake=gen(identity,meta)
   for image,is_fake in ((fake,True),(real,False)):
    score=F.normalize(sp(sar(image)),dim=1)@prototypes.T;one=(score.argmax(1)==label).sum().item();five=(score.topk(5,1).indices==label[:,None]).any(1).sum().item()
    if is_fake:fake1+=one;fake5+=five
    else:real1+=one;real5+=five
   total+=len(label)
 print({"samples":total,"generated_top1":fake1/total,"generated_top5":fake5/total,"real_top1":real1/total,"real_top5":real5/total})
if __name__=="__main__":main()
