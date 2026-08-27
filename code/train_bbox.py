from __future__ import annotations
import argparse,csv,json,random
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import numpy as np
from torch.nn import functional as F
from bbox_data import BBoxVehicleDataset
from bbox_models import MultiViewEncoder,ROIGenerator,ROIACDiscriminator
from sar_identity import SARIdentityEncoder
from train_sar_condition import Model as SARConditionModel
from rgb2sar.models import init_weights

def main():
 p=argparse.ArgumentParser(); p.add_argument("--rgb-root",type=Path,required=True);p.add_argument("--sar-root",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--epochs",type=int,default=20);p.add_argument("--epoch-size",type=int,default=2000);p.add_argument("--batch-size",type=int,default=16);p.add_argument("--workers",type=int,default=0);p.add_argument("--lr",type=float,default=2e-4);p.add_argument("--l1-weight",type=float,default=20);p.add_argument("--class-weight",type=float,default=2);p.add_argument("--identity-weight",type=float,default=10);p.add_argument("--condition-weight",type=float,default=5);p.add_argument("--band",default="all");p.add_argument("--polarization",default="all");p.add_argument("--depression",default="all");p.add_argument("--pre-cropped",action="store_true");p.add_argument("--rgb-encoder-checkpoint",type=Path);p.add_argument("--aligned-checkpoint",type=Path);p.add_argument("--condition-checkpoint",type=Path);p.add_argument("--freeze-rgb-encoder",action="store_true");p.add_argument("--device",default="cuda:0"); args=p.parse_args()
 args.output.mkdir(parents=True,exist_ok=True); ds=BBoxVehicleDataset(args.rgb_root,args.sar_root,epoch_size=args.epoch_size,band=args.band,polarization=args.polarization,depression=args.depression,pre_cropped=args.pre_cropped); dl=DataLoader(ds,args.batch_size,shuffle=True,num_workers=args.workers)
 dev=torch.device(args.device); enc=MultiViewEncoder().to(dev); gen=ROIGenerator().to(dev); disc=ROIACDiscriminator(len(ds.classes)).to(dev)
 for m in (enc,gen,disc):m.apply(init_weights)
 sar_identity=rgb_projection=sar_projection=None
 condition_model=None
 if args.aligned_checkpoint:
  aligned=torch.load(args.aligned_checkpoint,map_location=dev,weights_only=False);enc.load_state_dict(aligned["rgb_encoder"]);sar_identity=SARIdentityEncoder().to(dev);sar_identity.load_state_dict(aligned["sar_encoder"]);rgb_projection=nn.Linear(256,128).to(dev);sar_projection=nn.Linear(256,128).to(dev);rgb_projection.load_state_dict(aligned["rgb_projection"]);sar_projection.load_state_dict(aligned["sar_projection"])
  for model in (enc,sar_identity,rgb_projection,sar_projection):
   model.eval()
   for parameter in model.parameters():parameter.requires_grad=False
  args.freeze_rgb_encoder=True
 if args.condition_checkpoint:
  condition_model=SARConditionModel().to(dev);condition_model.load_state_dict(torch.load(args.condition_checkpoint,map_location=dev,weights_only=False)["model"]);condition_model.eval()
  for parameter in condition_model.parameters():parameter.requires_grad=False
 if args.rgb_encoder_checkpoint:
  pretrained=torch.load(args.rgb_encoder_checkpoint,map_location=dev,weights_only=False);enc.load_state_dict(pretrained["encoder"])
 if args.freeze_rgb_encoder:
  for parameter in enc.parameters():parameter.requires_grad=False
 generator_parameters=list(gen.parameters())+[p for p in enc.parameters() if p.requires_grad]
 og=torch.optim.Adam(generator_parameters,lr=args.lr,betas=(.5,.999));od=torch.optim.Adam(disc.parameters(),lr=args.lr,betas=(.5,.999)); ce=nn.CrossEntropyLoss();l1=nn.L1Loss()
 config={k:(str(v) if isinstance(v,Path) else v) for k,v in vars(args).items()};config["classes"]=ds.classes;(args.output/"config.json").write_text(json.dumps(config,indent=2),encoding="utf8")
 with (args.output/"history.csv").open("w",newline="") as f:csv.writer(f).writerow(["epoch","loss_g","loss_d","l1","class_acc","identity_cosine"])
 for epoch in range(1,args.epochs+1):
  totals=[0.,0.,0.,0.,0.,0.];bar=tqdm(dl,desc=f"bbox epoch {epoch}/{args.epochs}")
  for batch in bar:
   views,mask,meta,real,labels=[batch[k].to(dev) for k in ("views","view_mask","meta","roi","class_id")]
   if args.freeze_rgb_encoder:
    enc.eval()
    with torch.no_grad():identity,attention=enc(views,mask,meta)
   else:identity,attention=enc(views,mask,meta)
   fake=gen(identity,meta)
   od.zero_grad(set_to_none=True);dr,cr=disc(real);df,_=disc(fake.detach());loss_d=torch.relu(1-dr).mean()+torch.relu(1+df).mean()+ce(cr,labels);loss_d.backward();od.step()
   og.zero_grad(set_to_none=True);df,cf=disc(fake);recon=l1(fake,real);identity_loss=torch.zeros((),device=dev);identity_cosine=torch.zeros((),device=dev)
   if sar_identity is not None:
    rgb_z=F.normalize(rgb_projection(identity),dim=1);sar_z=F.normalize(sar_projection(sar_identity(fake)),dim=1);identity_cosine=(rgb_z*sar_z).sum(1).mean();identity_loss=1-identity_cosine
   condition_loss=torch.zeros((),device=dev)
   if condition_model is not None:
    az,dep,band,pol=condition_model(fake);target_dep=(meta[:,2]*4).round().long().sub(1).clamp(0,3);target_band=(1-meta[:,3]).long();target_pol=meta[:,4:8].argmax(1);condition_loss=(1-(az*meta[:,:2]).sum(1)).mean()+ce(dep,target_dep)+ce(band,target_band)+ce(pol,target_pol)
   loss_g=-df.mean()+args.l1_weight*recon+args.class_weight*ce(cf,labels)+args.identity_weight*identity_loss+args.condition_weight*condition_loss;loss_g.backward();og.step()
   acc=(cr.argmax(1)==labels).float().mean(); vals=[loss_g.item(),loss_d.item(),recon.item(),acc.item(),identity_cosine.item(),1];totals=[a+b for a,b in zip(totals,vals)];bar.set_postfix(g=f"{vals[0]:.2f}",d=f"{vals[1]:.2f}",id=f"{vals[4]:.2f}")
  n=totals[5]; row=[epoch,totals[0]/n,totals[1]/n,totals[2]/n,totals[3]/n,totals[4]/n]
  with (args.output/"history.csv").open("a",newline="") as f:csv.writer(f).writerow(row)
  ck={"epoch":epoch,"encoder":enc.state_dict(),"generator":gen.state_dict(),"args":config};torch.save(ck,args.output/"latest.pt")
  if epoch==1 or epoch%10==0:torch.save(ck,args.output/f"epoch_{epoch:04d}.pt");Image.fromarray(((fake[0,0].detach().cpu().clamp(-1,1).numpy()+1)*127.5).astype(np.uint8)).save(args.output/f"roi_{epoch:04d}.png")
if __name__=="__main__":main()
