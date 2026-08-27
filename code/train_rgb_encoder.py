from __future__ import annotations
import argparse,csv,json,math,random
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from bbox_data import load_multiview
from bbox_models import MultiViewEncoder

class RGBEpisodes(Dataset):
    def __init__(self,root:Path,episodes_per_class:int,train:bool):
        self.classes=sorted(p.name for p in root.iterdir() if p.is_dir());self.data=[load_multiview(root/c,64) for c in self.classes];self.episodes=episodes_per_class;self.train=train
    def __len__(self):return len(self.classes)*self.episodes
    def __getitem__(self,index):
        label=index//self.episodes; episode=index%self.episodes; views,available=self.data[label];views=views.clone();mask=available.clone()
        valid=torch.where(mask>0)[0]; query=int(valid[episode%len(valid)])
        if self.train:
            gain=random.uniform(.75,1.25);bias=random.uniform(-.12,.12);views=(views*gain+bias+torch.randn_like(views)*random.uniform(0,.04)).clamp(-1,1)
            # Drop up to half the available views, but always retain the queried direction.
            for v in valid.tolist():
                if v!=query and random.random()<.35:mask[v]=0
            if random.random()<.5:
                cut=random.randint(4,12);x=random.randint(0,64-cut);y=random.randint(0,64-cut);views[:, :, y:y+cut, x:x+cut]=0
        else:
            # Leave the queried view out: identity must come from the remaining viewpoints.
            if mask.sum()>1:mask[query]=0
        angle=math.radians(query*30);meta=torch.tensor([math.sin(angle),math.cos(angle),0,0,0,0,0,0,0,0],dtype=torch.float32)
        return views,mask,meta,label

def evaluate(enc,head,loader,device):
    enc.eval();head.eval();correct=total=0
    with torch.inference_mode():
        for views,mask,meta,label in loader:
            logits=head(enc(views.to(device),mask.to(device),meta.to(device))[0]);correct+=(logits.argmax(1).cpu()==label).sum().item();total+=len(label)
    return correct/total

def main():
    p=argparse.ArgumentParser();p.add_argument("--rgb-root",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--epochs",type=int,default=200);p.add_argument("--episodes-per-class",type=int,default=32);p.add_argument("--batch-size",type=int,default=64);p.add_argument("--lr",type=float,default=3e-4);p.add_argument("--device",default="cuda:0");a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    train=RGBEpisodes(a.rgb_root,a.episodes_per_class,True);val=RGBEpisodes(a.rgb_root,12,False);tl=DataLoader(train,a.batch_size,shuffle=True);vl=DataLoader(val,a.batch_size)
    dev=torch.device(a.device);enc=MultiViewEncoder().to(dev);head=nn.Linear(256,len(train.classes)).to(dev);opt=torch.optim.AdamW(list(enc.parameters())+list(head.parameters()),lr=a.lr,weight_decay=1e-4);ce=nn.CrossEntropyLoss();best=0
    with (a.output/"history.csv").open("w",newline="") as f:csv.writer(f).writerow(["epoch","loss","train_acc","leave_one_view_out_acc"])
    for epoch in range(1,a.epochs+1):
        enc.train();head.train();loss_sum=correct=total=0
        for views,mask,meta,label in tl:
            views,mask,meta,label=views.to(dev),mask.to(dev),meta.to(dev),label.to(dev);opt.zero_grad(set_to_none=True);embedding,_=enc(views,mask,meta);logits=head(embedding);loss=ce(logits,label);loss.backward();opt.step();loss_sum+=loss.item()*len(label);correct+=(logits.argmax(1)==label).sum().item();total+=len(label)
        val_acc=evaluate(enc,head,vl,dev);row=[epoch,loss_sum/total,correct/total,val_acc]
        with (a.output/"history.csv").open("a",newline="") as f:csv.writer(f).writerow(row)
        if val_acc>=best:
            best=val_acc;torch.save({"encoder":enc.state_dict(),"classifier":head.state_dict(),"classes":train.classes,"epoch":epoch,"val_acc":val_acc},a.output/"best.pt")
        if epoch==1 or epoch%10==0:print(row,flush=True)
    (a.output/"config.json").write_text(json.dumps({**vars(a),"rgb_root":str(a.rgb_root),"output":str(a.output),"classes":train.classes,"best_val_acc":best},indent=2),encoding="utf8")
if __name__=="__main__":main()
