from __future__ import annotations
import math, random, re
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from rgb2sar.data import rgba_to_rgb

SAR_RE = re.compile(r"^(X|KU)_(HH|HV|VH|VV)_(15|30|45|60)_(\d{1,3})_\d+$", re.I)
POLS = {"HH": 0, "HV": 1, "VH": 2, "VV": 3}

def read_annotation(xml_path: Path) -> tuple[tuple[int,int,int,int], dict[str, object]]:
    root = ET.parse(xml_path).getroot()
    def value(name: str) -> str:
        node = root.find(f".//{name}")
        if node is None or node.text is None: raise ValueError(f"missing {name} in {xml_path}")
        return node.text.strip().replace("°", "")
    bbox = tuple(int(float(value(k))) for k in ("xmin","ymin","xmax","ymax"))
    match = SAR_RE.match(xml_path.stem)
    if not match: raise ValueError(f"invalid SAR filename: {xml_path.name}")
    band, pol, dep, az = match.groups()
    return bbox, {"band": band.upper(), "pol": pol.upper(), "depression": int(dep), "azimuth": int(az)}

def metadata_vector(meta: dict[str, object], bbox: tuple[int,int,int,int]) -> torch.Tensor:
    az = math.radians(int(meta["azimuth"])); xmin,ymin,xmax,ymax=bbox
    pol = [0.0]*4; pol[POLS[str(meta["pol"])]]=1.0
    return torch.tensor([math.sin(az), math.cos(az), int(meta["depression"])/60.0,
        1.0 if meta["band"]=="X" else 0.0, *pol, (xmax-xmin)/128.0, (ymax-ymin)/128.0], dtype=torch.float32)

def image_tensor(image: Image.Image, size: int, rgb: bool) -> torch.Tensor:
    image=image.resize((size,size), Image.Resampling.BILINEAR).convert("RGB" if rgb else "L")
    arr=np.asarray(image,dtype=np.float32)
    if not rgb: arr=arr[:,:,None]
    return torch.from_numpy(arr.transpose(2,0,1).copy())/127.5-1.0

def load_multiview(folder: Path, size: int) -> tuple[torch.Tensor,torch.Tensor]:
    views=[]; mask=[]
    for index in range(1,13):
        path=folder/f"{index}.png"
        if path.exists():
            with Image.open(path) as im:
                im.thumbnail((size,size),Image.Resampling.LANCZOS); views.append(image_tensor(rgba_to_rgb(im),size,True)); mask.append(1.0)
        else: views.append(torch.zeros(3,size,size)); mask.append(0.0)
    return torch.stack(views),torch.tensor(mask,dtype=torch.float32)

class BBoxVehicleDataset(Dataset):
    def __init__(self,rgb_root:Path,sar_root:Path,rgb_size:int=64,roi_size:int=64,epoch_size:int=0,
                 band:str="all",polarization:str="all",depression:str="all",pre_cropped:bool=False):
        self.rgb_root,self.rgb_size,self.roi_size,self.pre_cropped=rgb_root,rgb_size,roi_size,pre_cropped; self.cache={}
        rgb_classes={p.name for p in rgb_root.iterdir() if p.is_dir()}; records=[]
        for tif in sar_root.rglob("*.tif"):
            if tif.parent.name not in rgb_classes or not tif.with_suffix(".xml").exists(): continue
            try: bbox,meta=read_annotation(tif.with_suffix(".xml"))
            except (ValueError,ET.ParseError): continue
            if band!="all" and meta["band"]!=band.upper(): continue
            if polarization!="all" and meta["pol"]!=polarization.upper(): continue
            if depression!="all" and meta["depression"]!=int(depression): continue
            records.append((tif,tif.parent.name,bbox,meta))
        if not records: raise RuntimeError("no valid RGB/SAR/XML records")
        self.records=records; self.classes=sorted({r[1] for r in records}); self.class_to_id={c:i for i,c in enumerate(self.classes)}
        self.random_epoch = 0 < epoch_size < len(records)
        self.epoch_size=epoch_size or len(records)
    def __len__(self): return self.epoch_size
    def _views(self,c):
        if c not in self.cache: self.cache[c]=load_multiview(self.rgb_root/c,self.rgb_size)
        return self.cache[c]
    def __getitem__(self,index):
        record_index = random.randrange(len(self.records)) if self.random_epoch else index % len(self.records)
        tif,c,bbox,meta=self.records[record_index]; xmin,ymin,xmax,ymax=bbox
        with Image.open(tif) as im:
            if self.pre_cropped: roi=image_tensor(im,self.roi_size,False);sar=image_tensor(im,128,False)
            else: sar=image_tensor(im,128,False);roi=image_tensor(im.crop((xmin,ymin,xmax,ymax)),self.roi_size,False)
        views,view_mask=self._views(c)
        return {"views":views,"view_mask":view_mask,"roi":roi,"sar":sar,"meta":metadata_vector(meta,bbox),
                "class_id":self.class_to_id[c],"class_name":c,"bbox":torch.tensor(bbox),"sar_path":str(tif)}
