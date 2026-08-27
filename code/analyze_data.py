from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
from rgb2sar.data import parse_sar
def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--rgb-root", type=Path, required=True); p.add_argument("--sar-root", type=Path, required=True); args = p.parse_args()
    rgb_counts = Counter(int(x.stem) for x in args.rgb_root.glob("*/*.png") if x.stem.isdigit())
    fields = {key: Counter() for key in ("band", "pol", "depression", "azimuth")}; invalid = 0
    for path in args.sar_root.rglob("*.tif"):
        meta = parse_sar(path)
        if meta is None: invalid += 1; continue
        for key in fields: fields[key][meta[key]] += 1
    print("RGB direction counts:", dict(sorted(rgb_counts.items())))
    print("SAR total:", sum(fields["azimuth"].values()), "invalid names:", invalid)
    for key, count in fields.items(): print(key, dict(sorted(count.items())))
if __name__ == "__main__": main()

