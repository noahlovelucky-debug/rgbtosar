from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from bbox_data import BBoxVehicleDataset,load_multiview
from bbox_models import MultiViewEncoder
from sar_identity import SARIdentityDataset,SARIdentityEncoder

def multi_positive_loss(a,b,labels,temp=.07):
 logits=a@b.T/temp; positive=labels[:,None].eq(labels[None,:]);la=-(logits.log_softmax(1)*positive).sum(1)/positive.sum(1);lb=-(logits.T.log_softmax(1)*positive.T).sum(1)/positive.T.sum(1);return (la.mean()+lb.mean())/2

def rgb_prototypes(enc,proj,rgb_root,classes,dev):
 result=[];enc.eval();proj.eval()
 with torch.inference_mode():
  for c in classes:
   views,mask=load_multiview(rgb_root/c,64);emb=[]
   for q in range(12):
    angle=math.radians(q*30);meta=torch.tensor([[math.sin(angle),math.cos(angle),0,0,0,0,0,0,0,0]],device=dev);identity,_=enc(views[None].to(dev),mask[None].to(dev),meta);emb.append(F.normalize(proj(identity),dim=1)[0])
   result.append(F.normalize(torch.stack(emb).mean(0),dim=0))
 return torch.stack(result)

def retrieval(sar,sp,test_loader,prototypes,dev):
 sar.eval();sp.eval();top1=top5=total=0
 with torch.inference_mode():
  for roi,label in test_loader:
   z=F.normalize(sp(sar(roi.to(dev))),dim=1);scores=z@prototypes.T;label=label.to(dev);top1+=(scores.argmax(1)==label).sum().item();top5+=(scores.topk(5,1).indices==label[:,None]).any(1).sum().item();total+=len(label)
 return top1/total,top5/total

def main():
 p=argparse.ArgumentParser();p.add_argument("--rgb-root",type=Path,required=True);p.add_argument("--sar-train-root",type=Path,required=True);p.add_argument("--sar-test-root",type=Path,required=True);p.add_argument("--rgb-checkpoint",type=Path,required=True);p.add_argument("--sar-checkpoint",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--epochs",type=int,default=5);p.add_argument("--batch-size",type=int,default=64);p.add_argument("--pre-cropped",action="store_true");p.add_argument("--device",default="cuda:0");a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);dev=torch.device(a.device)
 rgb_ck=torch.load(a.rgb_checkpoint,map_location=dev,weights_only=False);sar_ck=torch.load(a.sar_checkpoint,map_location=dev,weights_only=False);classes=rgb_ck["classes"];rgb=MultiViewEncoder().to(dev);rgb.load_state_dict(rgb_ck["encoder"]);sar=SARIdentityEncoder().to(dev);sar.load_state_dict(sar_ck["encoder"])
 for p0 in rgb.parameters():p0.requires_grad=False
 rp=nn.Linear(256,128).to(dev);sp=nn.Linear(256,128).to(dev);opt=torch.optim.AdamW(list(sar.parameters())+list(rp.parameters())+list(sp.parameters()),lr=1e-4,weight_decay=1e-4)
 train=BBoxVehicleDataset(a.rgb_root,a.sar_train_root,epoch_size=0,pre_cropped=a.pre_cropped);loader=DataLoader(train,a.batch_size,shuffle=True,num_workers=0);test=SARIdentityDataset(a.sar_test_root,classes,False,a.pre_cropped);test_loader=DataLoader(test,256,num_workers=8,persistent_workers=True)
 with (a.output/"history.csv").open("w",newline="") as f:csv.writer(f).writerow(["epoch","contrastive_loss","retrieval_top1","retrieval_top5"])
 best=0
 for epoch in range(1,a.epochs+1):
  sar.train();rp.train();sp.train();total_loss=steps=0
  for batch in loader:
   views,mask,meta,roi,label=[batch[k].to(dev) for k in ("views","view_mask","meta","roi","class_id")];opt.zero_grad(set_to_none=True)
   with torch.no_grad():rid,_=rgb(views,mask,meta)
   rz=F.normalize(rp(rid),dim=1);sz=F.normalize(sp(sar(roi)),dim=1);loss=multi_positive_loss(rz,sz,label);loss.backward();opt.step();total_loss+=loss.item();steps+=1
  prototypes=rgb_prototypes(rgb,rp,a.rgb_root,classes,dev);top1,top5=retrieval(sar,sp,test_loader,prototypes,dev);row=[epoch,total_loss/steps,top1,top5]
  with (a.output/"history.csv").open("a",newline="") as f:csv.writer(f).writerow(row)
  print(row,flush=True)
  if top1>=best:best=top1;torch.save({"rgb_encoder":rgb.state_dict(),"sar_encoder":sar.state_dict(),"rgb_projection":rp.state_dict(),"sar_projection":sp.state_dict(),"classes":classes,"epoch":epoch,"top1":top1,"top5":top5},a.output/"best.pt")
 (a.output/"config.json").write_text(json.dumps({**{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},"train_pairs":len(train),"test_samples":len(test),"best_top1":best},indent=2),encoding="utf8")
if __name__=="__main__":main()
