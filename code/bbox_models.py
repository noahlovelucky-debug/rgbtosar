from __future__ import annotations
import torch
from torch import nn
from rgb2sar.models import init_weights

class MultiViewEncoder(nn.Module):
    def __init__(self,dim=256):
        super().__init__(); self.cnn=nn.Sequential(nn.Conv2d(3,32,4,2,1),nn.ReLU(True),nn.Conv2d(32,64,4,2,1),nn.BatchNorm2d(64),nn.ReLU(True),nn.Conv2d(64,128,4,2,1),nn.BatchNorm2d(128),nn.ReLU(True),nn.AdaptiveAvgPool2d(1))
        self.project=nn.Linear(128,dim); self.view_embed=nn.Parameter(torch.randn(12,dim)*.02); self.query=nn.Sequential(nn.Linear(10,dim),nn.ReLU(),nn.Linear(dim,dim)); self.score=nn.Linear(dim,1)
    def forward(self,views,mask,meta):
        b,v,c,h,w=views.shape; tokens=self.project(self.cnn(views.reshape(b*v,c,h,w)).flatten(1)).reshape(b,v,-1)+self.view_embed[None]
        q=self.query(meta)[:,None,:]; logits=self.score(torch.tanh(tokens+q)).squeeze(-1); logits=logits.masked_fill(mask==0,-1e4)
        weights=logits.softmax(1); return (tokens*weights[:,:,None]).sum(1),weights

class ROIGenerator(nn.Module):
    def __init__(self,dim=256):
        super().__init__(); self.meta=nn.Sequential(nn.Linear(10,128),nn.ReLU(True)); self.fc=nn.Linear(dim+128,256*4*4)
        layers=[]; channels=256
        for out in (128,64,32,16): layers += [nn.ConvTranspose2d(channels,out,4,2,1),nn.BatchNorm2d(out),nn.ReLU(True)]; channels=out
        layers += [nn.Conv2d(channels,1,3,1,1),nn.Tanh()]; self.net=nn.Sequential(*layers)
    def forward(self,identity,meta): return self.net(self.fc(torch.cat([identity,self.meta(meta)],1)).reshape(-1,256,4,4))

class ROIACDiscriminator(nn.Module):
    def __init__(self,num_classes):
        super().__init__(); self.features=nn.Sequential(nn.Conv2d(1,32,4,2,1),nn.LeakyReLU(.2,True),nn.Conv2d(32,64,4,2,1),nn.LeakyReLU(.2,True),nn.Conv2d(64,128,4,2,1),nn.LeakyReLU(.2,True),nn.Conv2d(128,256,4,2,1),nn.LeakyReLU(.2,True))
        self.real=nn.Conv2d(256,1,4); self.classifier=nn.Linear(256,num_classes)
    def forward(self,x):
        f=self.features(x); return self.real(f).flatten(1),self.classifier(f.mean((2,3)))

