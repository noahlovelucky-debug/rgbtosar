from __future__ import annotations
import random
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset
from bbox_data import read_annotation,image_tensor

class SARIdentityDataset(Dataset):
    def __init__(self,root:Path,classes:list[str]|None=None,train:bool=False,pre_cropped:bool=False):
        found=sorted(p.name for p in root.iterdir() if p.is_dir());self.classes=classes or found;self.class_to_id={c:i for i,c in enumerate(self.classes)};self.train=train;self.pre_cropped=pre_cropped;self.records=[]
        for c in self.classes:
            folder=root/c
            if not folder.exists():continue
            for tif in folder.glob("*.tif"):
                xml=tif.with_suffix(".xml")
                if not xml.exists():continue
                try:bbox,meta=read_annotation(xml)
                except Exception:continue
                self.records.append((tif,bbox,self.class_to_id[c]))
    def __len__(self):return len(self.records)
    def __getitem__(self,index):
        tif,bbox,label=self.records[index]
        with Image.open(tif) as im:roi=image_tensor(im if self.pre_cropped else im.crop(bbox),64,False)
        if self.train:
            roi=(roi*random.uniform(.85,1.15)+random.uniform(-.08,.08)+torch.randn_like(roi)*random.uniform(0,.035)).clamp(-1,1)
            if random.random()<.3:
                cut=random.randint(3,9);x=random.randint(0,64-cut);y=random.randint(0,64-cut);roi[:,y:y+cut,x:x+cut]=0
        return roi,label

class SARIdentityEncoder(nn.Module):
    def __init__(self,dim=256):
        super().__init__();self.net=nn.Sequential(
            nn.Conv2d(1,32,3,1,1),nn.GroupNorm(8,32),nn.SiLU(),nn.Conv2d(32,32,4,2,1),nn.SiLU(),
            nn.Conv2d(32,64,3,1,1),nn.GroupNorm(8,64),nn.SiLU(),nn.Conv2d(64,64,4,2,1),nn.SiLU(),
            nn.Conv2d(64,128,3,1,1),nn.GroupNorm(16,128),nn.SiLU(),nn.Conv2d(128,128,4,2,1),nn.SiLU(),
            nn.Conv2d(128,256,3,1,1),nn.GroupNorm(32,256),nn.SiLU(),nn.AdaptiveAvgPool2d(1));self.project=nn.Linear(256,dim)
    def forward(self,x):return self.project(self.net(x).flatten(1))
