"""Build RGB_15 by segmentation, camera constraints and shared 3-D Gaussian rendering."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from collections import deque
from pathlib import Path

from prepare_rgb3d import prepare, run_sfm


def train_command(args: argparse.Namespace, rgba_root: Path, output: Path, gpu: str) -> tuple[list[str], dict[str, str]]:
    command = [sys.executable, str(Path(__file__).with_name("train_object_gsplat.py")),
               "--rgba-root", str(rgba_root), "--output", str(output),
               "--steps", str(args.steps), "--train-resolution", str(args.train_resolution),
               "--render-resolution", str(args.render_resolution),
               "--gaussians", str(args.gaussians), "--camera-steps", str(args.camera_steps),
               "--device", "cuda:0"]
    environment = os.environ.copy(); environment["CUDA_VISIBLE_DEVICES"] = gpu
    environment.setdefault("CUDA_HOME", str(Path(sys.executable).parent.parent))
    environment["PATH"] = str(Path(sys.executable).parent) + os.pathsep + environment.get("PATH", "")
    environment.setdefault("TORCH_CUDA_ARCH_LIST", "7.5")  # Quadro RTX 6000 in this workspace
    target_include = Path(sys.executable).parent.parent / "targets" / "x86_64-linux" / "include"
    environment["CPATH"] = str(target_include) + os.pathsep + environment.get("CPATH", "")
    return command, environment


def main() -> None:
    parser = argparse.ArgumentParser(description="Create validated 15-degree RGB via 3-D Gaussian splatting")
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--classes", nargs="*")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--gaussians", type=int, default=40000)
    parser.add_argument("--train-resolution", type=int, default=192)
    parser.add_argument("--render-resolution", type=int, default=512)
    parser.add_argument("--camera-steps", type=int, default=700)
    parser.add_argument("--prepare-resolution", type=int, default=768)
    parser.add_argument("--min-registered", type=int, default=8)
    parser.add_argument("--min-source-psnr", type=float, default=18.0)
    parser.add_argument("--min-alpha-iou", type=float, default=0.70)
    parser.add_argument("--min-novel-component", type=float, default=0.70)
    parser.add_argument("--min-novel-area-ratio", type=float, default=0.55)
    parser.add_argument("--max-novel-area-ratio", type=float, default=1.45)
    parser.add_argument("--skip-sfm", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    available = sorted(path for path in args.rgb_root.iterdir() if path.is_dir())
    if args.classes:
        wanted = set(args.classes); available = [path for path in available if path.name in wanted]
        missing = wanted - {path.name for path in available}
        if missing: raise RuntimeError(f"unknown classes: {sorted(missing)}")
    if not available: raise RuntimeError("no RGB class folders selected")
    args.output.mkdir(parents=True, exist_ok=True); args.work_root.mkdir(parents=True, exist_ok=True)

    jobs = deque()
    reports = {}
    for index, source in enumerate(available, 1):
        work = args.work_root / source.name
        print(f"prepare/cameras [{index}/{len(available)}] {source.name}", flush=True)
        manifest = prepare(source, work, args.prepare_resolution, 0.08, args.overwrite)
        camera_report = ({"success": False, "registered": 0, "required": args.min_registered,
                          "reason": "SfM skipped; constrained orbit camera will be optimised"}
                         if args.skip_sfm else run_sfm(work, min(args.min_registered, manifest["views"])))
        reports[source.name] = {"prepare": manifest, "camera": camera_report}
        model_output = work / "gsplat"
        validation_file = model_output / "cameras_and_renders.json"
        if args.overwrite or not validation_file.is_file():
            jobs.append((source.name, work / "rgba", model_output))

    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus: raise ValueError("--gpus cannot be empty")
    active: dict[str, tuple[subprocess.Popen, str, Path]] = {}
    while jobs or active:
        for gpu in gpus:
            if gpu in active or not jobs: continue
            class_name, rgba, model_output = jobs.popleft()
            command, environment = train_command(args, rgba, model_output, gpu)
            log_path = model_output.parent / "train.log"; log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("w", encoding="utf-8")
            print(f"GPU {gpu}: training {class_name}", flush=True)
            process = subprocess.Popen(command, env=environment, stdout=handle,
                                       stderr=subprocess.STDOUT, text=True)
            active[gpu] = (process, class_name, handle)
        if active:
            gpu = next(iter(active))
            process, class_name, handle = active[gpu]
            return_code = process.wait(timeout=None); handle.close(); del active[gpu]
            if return_code:
                raise RuntimeError(f"3-D Gaussian training failed for {class_name}; see "
                                   f"{args.work_root / class_name / 'train.log'}")
            print(f"GPU {gpu}: completed {class_name}", flush=True)

    rows = []
    failures = []
    for source in available:
        result_path = args.work_root / source.name / "gsplat" / "cameras_and_renders.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        psnr, iou = result["mean_source_psnr"], result["mean_source_alpha_iou"]
        component = result["min_novel_component_fraction"]
        min_area, max_area = result["min_novel_area_ratio"], result["max_novel_area_ratio"]
        accepted = (psnr >= args.min_source_psnr and iou >= args.min_alpha_iou
                    and component >= args.min_novel_component
                    and min_area >= args.min_novel_area_ratio
                    and max_area <= args.max_novel_area_ratio)
        reports[source.name]["gsplat"] = {"mean_source_psnr": psnr,
                                          "mean_source_alpha_iou": iou,
                                          "min_novel_component_fraction": component,
                                          "novel_area_ratio_range": [min_area, max_area],
                                          "accepted": accepted}
        if not accepted:
            failures.append(f"{source.name}: PSNR={psnr:.2f}, alpha IoU={iou:.3f}, "
                            f"novel component={component:.3f}, area=[{min_area:.3f},{max_area:.3f}]")
            continue
        destination = args.output / source.name; destination.mkdir(parents=True, exist_ok=True)
        for angle in range(0, 360, 15):
            source_render = args.work_root / source.name / "gsplat" / "renders" / f"{angle}.png"
            if not source_render.is_file(): raise RuntimeError(f"missing render: {source_render}")
            shutil.copy2(source_render, destination / source_render.name)
            rows.append({"class": source.name, "angle": angle, "method": "object_3d_gaussian",
                         "source_psnr": f"{psnr:.4f}", "source_alpha_iou": f"{iou:.6f}"})
    (args.work_root / "reconstruction_report.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    if failures:
        raise RuntimeError("Rejected reconstructions (not copied into RGB_15):\n" + "\n".join(failures))
    with (args.output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("class", "angle", "method",
                                                     "source_psnr", "source_alpha_iou"))
        writer.writeheader(); writer.writerows(rows)
    print(f"accepted {len(available)} classes and wrote {len(rows)} renders to {args.output}")


if __name__ == "__main__":
    main()
