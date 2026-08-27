from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from sar_identity import SARIdentityDataset,SARIdentityEncoder

def evaluate(enc,head,loader,dev):
 enc.eval();head.eval();correct=top5=total=0
 with torch.inference_mode():
  for x,y in loader:
   logits=head(enc(x.to(dev))); y=y.to(dev);correct+=(logits.argmax(1)==y).sum().item();top5+=(logits.topk(5,1).indices==y[:,None]).any(1).sum().item();total+=len(y)
 return correct/total,top5/total
def main():
 p=argparse.ArgumentParser();p.add_argument("--train-root",type=Path,required=True);p.add_argument("--test-root",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--epochs",type=int,default=20);p.add_argument("--batch-size",type=int,default=256);p.add_argument("--workers",type=int,default=8);p.add_argument("--pre-cropped",action="store_true");p.add_argument("--device",default="cuda:0");a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 train=SARIdentityDataset(a.train_root,train=True,pre_cropped=a.pre_cropped);test=SARIdentityDataset(a.test_root,train.classes,False,a.pre_cropped);tl=DataLoader(train,a.batch_size,shuffle=True,num_workers=a.workers,persistent_workers=a.workers>0,pin_memory=True);vl=DataLoader(test,a.batch_size,num_workers=a.workers,persistent_workers=a.workers>0,pin_memory=True)
 dev=torch.device(a.device);enc=SARIdentityEncoder().to(dev);head=nn.Linear(256,len(train.classes)).to(dev);opt=torch.optim.AdamW(list(enc.parameters())+list(head.parameters()),lr=3e-4,weight_decay=1e-4);ce=nn.CrossEntropyLoss(label_smoothing=.05);best=0
 with (a.output/"history.csv").open("w",newline="") as f:csv.writer(f).writerow(["epoch","loss","train_acc","test_top1","test_top5"])
 for epoch in range(1,a.epochs+1):
  enc.train();head.train();ls=correct=total=0
  for x,y in tl:
   x,y=x.to(dev,non_blocking=True),y.to(dev,non_blocking=True);opt.zero_grad(set_to_none=True);logits=head(enc(x));loss=ce(logits,y);loss.backward();opt.step();ls+=loss.item()*len(y);correct+=(logits.argmax(1)==y).sum().item();total+=len(y)
  top1,top5=evaluate(enc,head,vl,dev);row=[epoch,ls/total,correct/total,top1,top5]
  with (a.output/"history.csv").open("a",newline="") as f:csv.writer(f).writerow(row)
  print(row,flush=True)
  if top1>=best:best=top1;torch.save({"encoder":enc.state_dict(),"classifier":head.state_dict(),"classes":train.classes,"epoch":epoch,"test_top1":top1,"test_top5":top5},a.output/"best.pt")
 (a.output/"config.json").write_text(json.dumps({**{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},"train_samples":len(train),"test_samples":len(test),"best_top1":best},indent=2),encoding="utf8")
if __name__=="__main__":main()
