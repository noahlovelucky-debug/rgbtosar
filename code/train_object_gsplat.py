"""Masked turntable reconstruction with one explicit 3-D Gaussian model.

The 12 segmented views share a constrained orbit camera model.  All source
views optimise the same Gaussian cloud; the 15-degree images are then rendered
from virtual cameras.  This is intentionally separate from the rejected 2-D
affine interpolation baseline.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image
import cv2
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from gsplat import rasterization


def load_views(root: Path, resolution: int, device: torch.device):
    records = []
    for path in root.glob("*.png"):
        try:
            angle = int(path.stem)
        except ValueError:
            continue
        with Image.open(path) as opened:
            array = np.asarray(opened.convert("RGBA").resize(
                (resolution, resolution), Image.Resampling.LANCZOS), dtype=np.float32).copy() / 255.0
        records.append((angle, array))
    records.sort(key=lambda item: item[0])
    if len(records) < 6:
        raise RuntimeError(f"need at least six segmented views in {root}")
    images = torch.from_numpy(np.stack([item[1] for item in records])).to(device)
    angles = torch.tensor([item[0] for item in records], dtype=torch.float32, device=device)
    boxes, centres = [], []
    for alpha in images[..., 3]:
        ys, xs = torch.where(alpha > 0.25)
        if not len(xs):
            boxes.append((resolution * .5, resolution * .5))
            centres.append((resolution * .5, resolution * .5))
        else:
            boxes.append((float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)))
            centres.append((float(xs.float().mean()), float(ys.float().mean())))
    return images, angles, torch.tensor(boxes, device=device), torch.tensor(centres, device=device)


def orbit_matrices(angles: torch.Tensor, elevations: torch.Tensor,
                   radius: torch.Tensor) -> torch.Tensor:
    az = torch.deg2rad(angles)
    el = torch.deg2rad(elevations)
    origins = torch.stack((radius * torch.sin(az) * torch.cos(el),
                           radius * torch.sin(el),
                           radius * torch.cos(az) * torch.cos(el)), dim=-1)
    forward = F.normalize(-origins, dim=-1)
    world_up = torch.zeros_like(forward); world_up[:, 1] = 1
    right = F.normalize(torch.cross(forward, world_up, dim=-1), dim=-1)
    down = -F.normalize(torch.cross(right, forward, dim=-1), dim=-1)
    rotation = torch.stack((right, down, forward), dim=1)
    translation = -(rotation @ origins[..., None]).squeeze(-1)
    result = torch.eye(4, device=angles.device).repeat(len(angles), 1, 1)
    result[:, :3, :3] = rotation
    result[:, :3, 3] = translation
    return result


def camera_intrinsics(angles: torch.Tensor, boxes: torch.Tensor, centres: torch.Tensor,
                      resolution: int, radius: float = 3.0) -> torch.Tensor:
    # Estimate per-frame focal length from its foreground extent.  This absorbs
    # the large source-camera distance differences that a single focal cannot.
    az = torch.deg2rad(angles)
    horizontal_half_extent = .78 * torch.abs(torch.cos(az)) + 1.35 * torch.abs(torch.sin(az))
    focal = (boxes[:, 0] * radius / (2 * horizontal_half_extent)).clamp(
        resolution * .55, resolution * 1.8)
    result = torch.zeros(len(angles), 3, 3, device=angles.device)
    result[:, 0, 0] = focal; result[:, 1, 1] = focal
    result[:, 0, 2] = centres[:, 0]; result[:, 1, 2] = centres[:, 1]
    result[:, 2, 2] = 1
    return result


def make_camera_state(angles: torch.Tensor, boxes: torch.Tensor, centres: torch.Tensor,
                      resolution: int) -> nn.ParameterDict:
    """Per-image camera variables, anchored to the known turntable angles.

    The source filenames provide a very good azimuth prior, but the original
    RGB captures are not guaranteed to have exactly the same focal length,
    crop, elevation, or distance.  These are deliberately small corrections,
    not 12 unconstrained poses: otherwise a Gaussian cloud can memorise a
    different object in each camera.
    """
    initial_K = camera_intrinsics(angles, boxes, centres, resolution)
    return nn.ParameterDict({
        "azimuth_delta": nn.Parameter(torch.zeros_like(angles)),
        "elevation": nn.Parameter(torch.full_like(angles, 10.0)),
        "log_radius": nn.Parameter(torch.full_like(angles, math.log(3.0))),
        "log_focal": nn.Parameter(initial_K[:, 0, 0].log()),
        "principal_delta": nn.Parameter(torch.zeros(len(angles), 2, device=angles.device)),
    })


def cameras_from_state(angles: torch.Tensor, boxes: torch.Tensor, centres: torch.Tensor,
                       resolution: int, state: nn.ParameterDict):
    calibrated_angles = angles + state["azimuth_delta"]
    radii = state["log_radius"].exp()
    viewmats = orbit_matrices(calibrated_angles, state["elevation"], radii)
    intrinsics = torch.zeros(len(angles), 3, 3, device=angles.device)
    focal = state["log_focal"].exp().clamp(resolution * .45, resolution * 2.2)
    intrinsics[:, 0, 0] = focal; intrinsics[:, 1, 1] = focal
    intrinsics[:, 0, 2] = centres[:, 0] + state["principal_delta"][:, 0]
    intrinsics[:, 1, 2] = centres[:, 1] + state["principal_delta"][:, 1]
    intrinsics[:, 2, 2] = 1
    return viewmats, intrinsics


def ellipsoid_proxy(count: int, device: torch.device) -> nn.ParameterDict:
    """A deliberately low-capacity silhouette proxy used only for calibration."""
    indices = torch.arange(count, device=device, dtype=torch.float32) + .5
    y = 1 - 2 * indices / count
    radial = (1 - y.square()).sqrt()
    phi = indices * math.pi * (3 - math.sqrt(5))
    unit = torch.stack((radial * torch.cos(phi), y, radial * torch.sin(phi)), dim=-1)
    extents = nn.Parameter(torch.tensor((.78, .50, 1.35), device=device))
    # The sigmoid parameterisation keeps the proxy a valid vehicle-sized body.
    return nn.ParameterDict({"unit": nn.Parameter(unit, requires_grad=False), "extents": extents})


def calibrate_cameras(angles: torch.Tensor, boxes: torch.Tensor, centres: torch.Tensor,
                      images: torch.Tensor, resolution: int, steps: int):
    """Fit camera nuisance parameters against all foreground silhouettes.

    This stage is independent of RGB texture.  It cannot fabricate geometry,
    but it eliminates the largest source of false geometry: treating unequal
    crops and elevations as one perfect 30-degree orbit.
    """
    state = make_camera_state(angles, boxes, centres, resolution)
    proxy = ellipsoid_proxy(5000, images.device)
    optimiser = torch.optim.Adam([
        {"params": state.parameters(), "lr": 2e-3},
        {"params": proxy["extents"], "lr": 5e-3},
    ])
    target_alpha = images[..., 3:4]
    initial_focal = camera_intrinsics(angles, boxes, centres, resolution)[:, 0, 0].log().detach()
    for step in range(steps):
        viewmats, intrinsics = cameras_from_state(angles, boxes, centres, resolution, state)
        extents = proxy["extents"].clamp(.15, 2.5)
        means = proxy["unit"] * extents
        rendered, alpha, _ = rasterization(
            means, torch.tensor([1, 0, 0, 0], device=images.device, dtype=images.dtype).repeat(len(means), 1),
            torch.full((len(means), 3), .028, device=images.device),
            torch.full((len(means),), .035, device=images.device),
            torch.ones(len(means), 3, device=images.device), viewmats, intrinsics,
            resolution, resolution, backgrounds=torch.zeros(len(images), 3, device=images.device), packed=False,
            rasterize_mode="antialiased")
        # Downweight RGB-free pixels; the contour itself has the useful signal.
        silhouette = F.binary_cross_entropy(alpha.clamp(1e-4, 1 - 1e-4), target_alpha)
        prior = (state["azimuth_delta"] / 4).square().mean()
        prior = prior + ((state["elevation"] - 10) / 8).square().mean()
        prior = prior + ((state["log_radius"] - math.log(3.0)) / .18).square().mean()
        prior = prior + ((state["log_focal"] - initial_focal) / .22).square().mean()
        prior = prior + (state["principal_delta"] / (resolution * .06)).square().mean()
        shape_prior = ((extents - extents.new_tensor((.78, .50, 1.35))) /
                       extents.new_tensor((.45, .35, .65))).square().mean()
        loss = silhouette + .08 * prior + .03 * shape_prior
        optimiser.zero_grad(set_to_none=True); loss.backward(); optimiser.step()
    with torch.no_grad():
        viewmats, intrinsics = cameras_from_state(angles, boxes, centres, resolution, state)
        report = {"azimuth_delta_deg": state["azimuth_delta"].detach().cpu().tolist(),
                  "elevation_deg": state["elevation"].detach().cpu().tolist(),
                  "radius": state["log_radius"].exp().detach().cpu().tolist(),
                  "focal_px": state["log_focal"].exp().detach().cpu().tolist(),
                  "principal_delta_px": state["principal_delta"].detach().cpu().tolist(),
                  "proxy_extents": proxy["extents"].clamp(.15, 2.5).detach().cpu().tolist(),
                  "final_silhouette_bce": float(silhouette.detach())}
    return state, viewmats.detach(), intrinsics.detach(), report


def visual_hull_points(count: int, images: torch.Tensor, viewmats: torch.Tensor,
                       intrinsics: torch.Tensor, device: torch.device,
                       grid_size: int = 80) -> torch.Tensor:
    """Carve a shared foreground volume and return its boundary points."""
    x = torch.linspace(-.88, .88, grid_size, device=device)
    y = torch.linspace(-.62, .62, grid_size, device=device)
    z = torch.linspace(-1.48, 1.48, grid_size, device=device)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    points = torch.stack((xx, yy, zz), dim=-1).reshape(-1, 3)
    # Small dilation tolerates imperfect focal/elevation estimates while still
    # requiring agreement from most of the 12 silhouettes.
    masks = F.max_pool2d(images[..., 3].unsqueeze(1), 9, stride=1, padding=4)
    votes = torch.zeros(len(points), dtype=torch.int16, device=device)
    homogeneous = torch.cat((points, torch.ones(len(points), 1, device=device)), dim=1)
    for view in range(len(images)):
        camera = homogeneous @ viewmats[view, :3].T
        pixel_h = camera @ intrinsics[view].T
        pixel = pixel_h[:, :2] / pixel_h[:, 2:].clamp_min(1e-5)
        grid = pixel.clone()
        grid[:, 0] = grid[:, 0] / max(1, images.shape[2] - 1) * 2 - 1
        grid[:, 1] = grid[:, 1] / max(1, images.shape[1] - 1) * 2 - 1
        sampled = F.grid_sample(masks[view:view + 1], grid[None, None],
                                mode="bilinear", align_corners=True)[0, 0, 0]
        votes += (sampled > .15).to(votes.dtype)
    required = max(4, math.ceil(len(images) * .58))
    occupied = (votes >= required).reshape(grid_size, grid_size, grid_size)
    neighbours = F.conv3d(occupied.float()[None, None], torch.ones(1, 1, 3, 3, 3, device=device),
                          padding=1)[0, 0]
    surface = occupied & (neighbours < 26.5)
    candidates = torch.stack((xx[surface], yy[surface], zz[surface]), dim=-1)
    if not len(candidates):
        raise RuntimeError("visual-hull carving produced no common vehicle volume")
    indices = torch.randint(len(candidates), (count,), device=device)
    spacing = max(1.76, 1.24, 2.96) / (grid_size - 1)
    return candidates[indices] + torch.randn(count, 3, device=device) * spacing * .15


def initialise_gaussians(count: int, images: torch.Tensor, viewmats: torch.Tensor,
                         intrinsics: torch.Tensor, device: torch.device):
    means = visual_hull_points(count, images, viewmats, intrinsics, device)
    extents = torch.tensor((.78, .50, 1.35), device=device)

    homogeneous = torch.cat((means, torch.ones(count, 1, device=device)), dim=1)
    camera = torch.einsum("vij,nj->vni", viewmats[:, :3], homogeneous)
    pixels_h = torch.einsum("vij,vnj->vni", intrinsics, camera)
    pixels = pixels_h[..., :2] / pixels_h[..., 2:].clamp_min(1e-5)
    normal = F.normalize(means / extents.square(), dim=-1)
    origins = torch.linalg.inv(viewmats)[:, :3, 3]
    facing = torch.einsum("nd,vnd->vn", normal,
                          F.normalize(origins[:, None] - means[None], dim=-1))
    colours = torch.full((count, 3), .45, device=device)
    best = facing.argmax(0)
    for view in range(len(images)):
        selected = best == view
        if not selected.any():
            continue
        xy = pixels[view, selected]
        grid = xy.clone()
        grid[:, 0] = grid[:, 0] / max(1, images.shape[2] - 1) * 2 - 1
        grid[:, 1] = grid[:, 1] / max(1, images.shape[1] - 1) * 2 - 1
        sampled = F.grid_sample(images[view:view + 1].permute(0, 3, 1, 2),
                                grid[None, None], align_corners=True)[0, :, 0].T
        valid = sampled[:, 3] > .08
        selected_indices = torch.where(selected)[0]
        colours[selected_indices[valid]] = sampled[valid, :3]
    colours = colours.clamp(.01, .99)
    params = nn.ParameterDict({
        "means": nn.Parameter(means),
        "scales": nn.Parameter(torch.full((count, 3), math.log(.032 * math.sqrt(25000 / count)), device=device)),
        "quats": nn.Parameter(F.normalize(torch.randn(count, 4, device=device), dim=-1)),
        "opacities": nn.Parameter(torch.zeros(count, device=device)),
        "colors": nn.Parameter(torch.logit(colours)),
    })
    return params


def render(params, viewmats, intrinsics, resolution, backgrounds=None, colours_override=None):
    colours = torch.sigmoid(params["colors"]) if colours_override is None else colours_override
    rendered, alpha, info = rasterization(
        params["means"], params["quats"], torch.exp(params["scales"]),
        torch.sigmoid(params["opacities"]), colours, viewmats, intrinsics,
        resolution, resolution, near_plane=.5, far_plane=6.0,
        backgrounds=backgrounds, packed=False,
        rasterize_mode="antialiased")
    return rendered, alpha, info


def projective_colours(params, target_angle: int, images: torch.Tensor, source_angles: torch.Tensor,
                       source_viewmats: torch.Tensor, source_intrinsics: torch.Tensor) -> torch.Tensor:
    """Fill the shared 3-D surface texture by projecting nearby source views."""
    means = params["means"]
    homogeneous = torch.cat((means, torch.ones(len(means), 1, device=means.device)), dim=1)
    normal = F.normalize(means / means.new_tensor((.78 ** 2, .50 ** 2, 1.35 ** 2)), dim=-1)
    origins = torch.linalg.inv(source_viewmats)[:, :3, 3]
    weighted = torch.zeros(len(means), 3, device=means.device)
    weight_sum = torch.zeros(len(means), 1, device=means.device)
    for view, source_angle in enumerate(source_angles):
        camera = homogeneous @ source_viewmats[view, :3].T
        pixel_h = camera @ source_intrinsics[view].T
        pixel = pixel_h[:, :2] / pixel_h[:, 2:].clamp_min(1e-5)
        grid = pixel.clone()
        grid[:, 0] = grid[:, 0] / max(1, images.shape[2] - 1) * 2 - 1
        grid[:, 1] = grid[:, 1] / max(1, images.shape[1] - 1) * 2 - 1
        rgba = F.grid_sample(images[view:view + 1].permute(0, 3, 1, 2), grid[None, None],
                             mode="bilinear", padding_mode="zeros", align_corners=True)[0, :, 0].T
        difference = abs(float(source_angle) - target_angle) % 360
        difference = min(difference, 360 - difference)
        angular = math.exp(-0.5 * (difference / 24.0) ** 2)
        facing = (normal * F.normalize(origins[view] - means, dim=-1)).sum(-1).clamp_min(0)
        weight = rgba[:, 3:4] * facing[:, None].pow(.5) * angular
        weighted += rgba[:, :3] * weight; weight_sum += weight
    fallback = torch.sigmoid(params["colors"])
    return torch.where(weight_sum > 1e-4, weighted / weight_sum.clamp_min(1e-4), fallback).clamp(0, 1)


def interpolate_camera(angle: int, source_angles: list[int], viewmats: torch.Tensor,
                       intrinsics: torch.Tensor, camera_state: nn.ParameterDict,
                       render_resolution: int, train_resolution: int):
    left = max(source_angles, key=lambda value: -((angle - value) % 360))
    right = min(source_angles, key=lambda value: (value - angle) % 360)
    li, ri = source_angles.index(left), source_angles.index(right)
    span = (right - left) % 360
    fraction = 0 if span == 0 else ((angle - left) % 360) / span
    # Construct the exact virtual 15-degree azimuth, while linearly
    # interpolating the *estimated* nuisance camera parameters from its two
    # neighbouring photographs.  This is an orbit model, not 2-D warping.
    azimuth_delta = ((1 - fraction) * camera_state["azimuth_delta"][li]
                     + fraction * camera_state["azimuth_delta"][ri])
    elevation = ((1 - fraction) * camera_state["elevation"][li]
                 + fraction * camera_state["elevation"][ri]).reshape(1)
    log_radius = ((1 - fraction) * camera_state["log_radius"][li]
                  + fraction * camera_state["log_radius"][ri]).reshape(1)
    matrix = orbit_matrices(viewmats.new_tensor([angle]) + azimuth_delta.reshape(1),
                            elevation, log_radius.exp())
    K = (1 - fraction) * intrinsics[li:li + 1] + fraction * intrinsics[ri:ri + 1]
    scale = render_resolution / train_resolution
    K = K.clone(); K[:, :2] *= scale
    return matrix, K


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgba-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--gaussians", type=int, default=40000)
    parser.add_argument("--train-resolution", type=int, default=192)
    parser.add_argument("--render-resolution", type=int, default=512)
    parser.add_argument("--camera-steps", type=int, default=700,
                        help="silhouette-only camera calibration iterations before 3-D fitting")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device); args.output.mkdir(parents=True, exist_ok=True)
    images, angles, boxes, centres = load_views(args.rgba_root, args.train_resolution, device)
    camera_state, viewmats, intrinsics, camera_report = calibrate_cameras(
        angles, boxes, centres, images, args.train_resolution, args.camera_steps)
    (args.output / "estimated_cameras.json").write_text(
        json.dumps(camera_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("camera calibration", json.dumps(camera_report, ensure_ascii=False), flush=True)
    params = initialise_gaussians(args.gaussians, images, viewmats, intrinsics, device)
    initial_means = params["means"].detach().clone()
    learning_rates = {"means": 3e-6, "scales": 1e-4, "quats": 5e-5,
                      "opacities": 1e-3, "colors": 8e-3}
    optimizers = {key: torch.optim.Adam([params[key]], lr=value, eps=1e-15)
                  for key, value in learning_rates.items()}
    progress = tqdm(range(args.steps), desc=f"3-D Gaussians {args.rgba_root.parent.name}")
    for step in progress:
        index = random.randrange(len(images))
        background = torch.rand(1, 3, device=device) * .15
        rgb, alpha, info = render(params, viewmats[index:index + 1],
                                  intrinsics[index:index + 1], args.train_resolution,
                                  background)
        target_alpha = images[index:index + 1, ..., 3:4]
        target_rgb = images[index:index + 1, ..., :3] * target_alpha + background[:, None, None] * (1 - target_alpha)
        foreground = 1 + 2 * target_alpha
        rgb_loss = ((rgb - target_rgb).abs() * foreground).mean()
        alpha_loss = F.binary_cross_entropy(alpha.clamp(1e-5, 1 - 1e-5), target_alpha)
        geometry_loss = (params["means"] - initial_means).square().mean()
        oversized = torch.exp(params["scales"]).sub(.055).clamp_min(0).square().mean()
        opacity_prior = (torch.sigmoid(params["opacities"]) - .55).square().mean()
        loss = (rgb_loss + .35 * alpha_loss + 2.0 * geometry_loss
                + .2 * oversized + .05 * opacity_prior)
        for optimizer in optimizers.values(): optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for optimizer in optimizers.values(): optimizer.step()
        if step % 25 == 0:
            progress.set_postfix(loss=f"{loss.item():.4f}", count=len(params["means"]))

    torch.save({"params": params.state_dict(), "angles": angles.cpu(),
                "viewmats": viewmats.cpu(), "intrinsics": intrinsics.cpu(),
                "camera_state": {key: value.detach().cpu() for key, value in camera_state.items()},
                "camera_calibration": camera_report, "args": vars(args)}, args.output / "latest.pt")
    source_angles = [int(value) for value in angles.tolist()]
    render_dir = args.output / "renders"; render_dir.mkdir(exist_ok=True)
    validation, manifest, rendered_masks = [], [], {}
    with torch.inference_mode():
        for angle in range(0, 360, 15):
            matrix, K = interpolate_camera(angle, source_angles, viewmats, intrinsics, camera_state,
                                           args.render_resolution, args.train_resolution)
            texture = projective_colours(params, angle, images, angles, viewmats, intrinsics)
            rgb, alpha, _ = render(params, matrix, K, args.render_resolution,
                                   torch.zeros(1, 3, device=device), texture)
            array = torch.cat((rgb[0], alpha[0]), dim=-1).clamp(0, 1).cpu().numpy()
            rendered_masks[angle] = array[..., 3] > .5
            output_path = render_dir / f"{angle}.png"
            Image.fromarray((array * 255 + .5).astype(np.uint8), "RGBA").save(output_path)
            manifest.append({"angle": angle, "file": str(output_path)})
            if angle in source_angles:
                target = F.interpolate(images[source_angles.index(angle)].permute(2, 0, 1)[None],
                                       (args.render_resolution,) * 2, mode="bilinear", align_corners=False)[0]
                predicted = torch.from_numpy(array).to(device).permute(2, 0, 1)
                mse = ((predicted[:3] * predicted[3:] - target[:3] * target[3:]).square().mean()
                       + (predicted[3:] - target[3:]).square().mean()).item() / 2
                pm, tm = predicted[3] > .5, target[3] > .5
                iou = (pm & tm).sum().item() / max(1, (pm | tm).sum().item())
                validation.append({"angle": angle, "psnr": -10 * math.log10(max(mse, 1e-10)),
                                   "alpha_iou": iou})
    novel_geometry = []
    for angle in range(15, 360, 30):
        mask = rendered_masks[angle].astype(np.uint8)
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        foreground = int(mask.sum())
        largest = int(stats[1:, cv2.CC_STAT_AREA].max()) if count > 1 else 0
        left, right = (angle - 15) % 360, (angle + 15) % 360
        reference_area = .5 * (rendered_masks[left].sum() + rendered_masks[right].sum())
        novel_geometry.append({"angle": angle,
                               "largest_component_fraction": largest / max(1, foreground),
                               "neighbour_area_ratio": foreground / max(1, reference_area)})
    report = {"method": "masked_constrained_orbit_3d_gaussian_splatting",
              "camera_calibration": camera_report,
              "source_angles": source_angles, "renders": manifest,
              "source_view_validation": validation,
              "mean_source_psnr": float(np.mean([v["psnr"] for v in validation])),
              "mean_source_alpha_iou": float(np.mean([v["alpha_iou"] for v in validation])),
              "novel_view_geometry": novel_geometry,
              "min_novel_component_fraction": min(v["largest_component_fraction"] for v in novel_geometry),
              "min_novel_area_ratio": min(v["neighbour_area_ratio"] for v in novel_geometry),
              "max_novel_area_ratio": max(v["neighbour_area_ratio"] for v in novel_geometry),
              "gaussian_count": len(params["means"])}
    (args.output / "cameras_and_renders.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
