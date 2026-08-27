"""Resumable extractor for SOC_40classes_cut.zip (safe after an interrupted unzip)."""
from __future__ import annotations
import argparse
from pathlib import Path
from zipfile import ZipFile

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("zip", type=Path, nargs="?", default=Path(r"Z:\amplitude 8-bit data_地距幅度8位数据.7z\SOC_40classes_cut.zip"))
    p.add_argument("destination", type=Path, nargs="?", default=Path(r"Z:\amplitude 8-bit data_地距幅度8位数据.7z"))
    args = p.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.zip) as archive:
        members = archive.infolist()
        for index, member in enumerate(members, 1):
            target = args.destination / member.filename
            if member.is_dir(): target.mkdir(parents=True, exist_ok=True); continue
            if target.is_file() and target.stat().st_size == member.file_size: continue
            archive.extract(member, args.destination)
            if index % 1000 == 0: print(f"{index}/{len(members)}", flush=True)
    print(f"complete: {len(members)} archive entries", flush=True)
if __name__ == "__main__": main()
