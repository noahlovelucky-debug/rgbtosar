from __future__ import annotations
import argparse,csv,math,re
from pathlib import Path
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from bbox_data import image_tensor,SAR_RE,POLS
from sar_identity import SARIdentityEncoder

class Data(Dataset):
 def __init__(self,root):
  self.files=list(Path(root).glob("*/*.tif"))
 def __len__(self):return len(self.files)
 def __getitem__(self,i):
  p=self.files[i];band,pol,dep,az=SAR_RE.match(p.stem).groups();r=math.radians(int(az))
  with Image.open(p) as im:x=image_tensor(im,64,False)
  return x,torch.tensor([math.sin(r),math.cos(r)]),[15,30,45,60].index(int(dep)),0 if band.upper()=="X" else 1,POLS[pol.upper()]
class Model(nn.Module):
 def __init__(self):
  super().__init__();self.encoder=SARIdentityEncoder();self.az=nn.Linear(256,2);self.dep=nn.Linear(256,4);self.band=nn.Linear(256,2);self.pol=nn.Linear(256,4)
 def forward(self,x):
  z=self.encoder(x);return F.normalize(self.az(z),dim=1),self.dep(z),self.band(z),self.pol(z)
from torch.nn import functional as F
def main():
 p=argparse.ArgumentParser();p.add_argument("--train-root",type=Path,required=True);p.add_argument("--test-root",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--epochs",type=int,default=10);p.add_argument("--device",default="cuda:0");a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);tr,te=Data(a.train_root),Data(a.test_root);tl=DataLoader(tr,256,shuffle=True,num_workers=8,persistent_workers=True);vl=DataLoader(te,256,num_workers=8,persistent_workers=True);dev=torch.device(a.device);m=Model().to(dev);opt=torch.optim.AdamW(m.parameters(),3e-4);ce=nn.CrossEntropyLoss();best=0
 with (a.output/"history.csv").open("w",newline="") as f:csv.writer(f).writerow(["epoch","az_cos","dep_acc","band_acc","pol_acc"])
 for e in range(1,a.epochs+1):
  m.train()
  for x,az,dep,band,pol in tl:
   x,az,dep,band,pol=x.to(dev),az.float().to(dev),dep.to(dev),band.to(dev),pol.to(dev);o=m(x);loss=(1-(o[0]*az).sum(1)).mean()+ce(o[1],dep)+ce(o[2],band)+ce(o[3],pol);opt.zero_grad();loss.backward();opt.step()
  m.eval();s=[0.,0.,0.,0.,0.]
  with torch.inference_mode():
   for x,az,dep,band,pol in vl:
    x,az=x.to(dev),az.float().to(dev);dep,band,pol=dep.to(dev),band.to(dev),pol.to(dev);o=m(x);n=len(dep);s[0]+=(o[0]*az).sum().item();s[1]+=(o[1].argmax(1)==dep).sum().item();s[2]+=(o[2].argmax(1)==band).sum().item();s[3]+=(o[3].argmax(1)==pol).sum().item();s[4]+=n
  row=[e,*[v/s[4] for v in s[:4]]];print(row,flush=True)
  with (a.output/"history.csv").open("a",newline="") as f:csv.writer(f).writerow(row)
  if row[1]>best:best=row[1];torch.save({"model":m.state_dict(),"epoch":e,"metrics":row[1:]},a.output/"best.pt")
if __name__=="__main__":main()
