"""FACT-SAR primitives for unpaired cross-view RGB-to-SAR synthesis.

FACT-SAR deliberately does not use a frozen native SAR classifier as a
generator teacher.  Instead, a generated SAR support set and an independent
real SAR support set are compared by the *updates* that they induce in a
random, frozen task learner.  The update is factorized into vehicle-identity
and acquisition-condition components.  A real query from the same vehicle at
a different acquisition condition anchors the identity component.

There is no RGB/SAR pixel correspondence anywhere in this module.  RGB and
SAR meet only through class identity and requested SAR acquisition metadata.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import random
from pathlib import Path

from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from bbox_data import image_tensor, metadata_vector
from joint_data import JointROIDataset
from saratrx import SOC40_CLASSES


FACT_ARCHITECTURE = "fact_sar_conditioned_v1"
ACQUISITION_TARGET_NAMES = ("band", "polarization", "depression", "azimuth")


def acquisition_targets(meta: torch.Tensor, depression: torch.Tensor,
                        azimuth: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Convert target SAR metadata to the four task-learner labels.

    The convention matches ``SARClassifier64``: X is 0, KU is 1; polarization
    is HH/HV/VH/VV; depression is 15/30/45/60; azimuth is a circular 12-bin
    variable.  These are targets, never inputs to the random learner.
    """
    if meta.ndim != 2 or meta.shape[1] < 8:
        raise ValueError(f"expected metadata [B,>=8], got {tuple(meta.shape)}")
    band = (1 - meta[:, 3].round().long()).clamp(0, 1)
    polarization = meta[:, 4:8].argmax(1)
    depression_bin = torch.round(depression.float() / 15.0).long().sub(1).clamp(0, 3)
    azimuth_bin = ((azimuth.long() + 15) % 360) // 30
    return band, polarization, depression_bin, azimuth_bin


def acquisition_code(meta: torch.Tensor, depression: torch.Tensor,
                     azimuth: torch.Tensor) -> torch.Tensor:
    """Return a stable joint band/pol/depression/azimuth code."""
    band, polarization, depression_bin, azimuth_bin = acquisition_targets(
        meta, depression, azimuth)
    return (((band * 4 + polarization) * 4 + depression_bin) * 12 + azimuth_bin)


def _record_name(record: tuple, root: Path) -> str:
    return str(Path(record[0]).relative_to(root))


def _split_counts(count: int, influence_fraction: float,
                  audit_fraction: float) -> tuple[int, int]:
    """Choose disjoint audit/influence counts while leaving realism data."""
    if count <= 1:
        return 0, 0
    audit = int(round(count * audit_fraction)) if audit_fraction else 0
    influence = int(round(count * influence_fraction)) if influence_fraction else 0
    if audit_fraction:
        audit = max(1, audit)
    if influence_fraction and count - audit >= 2:
        influence = max(1, influence)
    audit = min(audit, max(0, count - 1))
    influence = min(influence, max(0, count - audit - 1))
    return audit, influence


def split_fact_records(records: list[tuple], root: Path, manifest: Path,
                       influence_fraction: float, audit_fraction: float,
                       seed: int, filters: dict[str, str]
                       ) -> tuple[set[str], set[str], set[str]]:
    """Build a deterministic, stratified three-way SAR split.

    ``realism`` is the only split visible to the GAN discriminator;
    ``influence`` is the only split read by FACT episodes; ``audit`` is only
    used for development-time previews and metrics.  The official test root
    is not accepted by this API and therefore cannot enter the split.
    """
    if not 0.0 < influence_fraction < 1.0:
        raise ValueError("influence_fraction must be in (0, 1)")
    if not 0.0 < audit_fraction < 1.0:
        raise ValueError("audit_fraction must be in (0, 1)")
    if influence_fraction + audit_fraction >= 1.0:
        raise ValueError("influence_fraction + audit_fraction must be below one")
    record_names = sorted(_record_name(record, root) for record in records)
    record_digest = hashlib.sha256("\n".join(record_names).encode("utf-8")).hexdigest()
    expected = {
        "version": FACT_ARCHITECTURE,
        "source_root": str(root.resolve()),
        "record_sha256": record_digest,
        "filters": filters,
        "seed": seed,
        "influence_fraction": influence_fraction,
        "audit_fraction": audit_fraction,
    }
    if manifest.is_file():
        try:
            saved = json.loads(manifest.read_text(encoding="utf-8"))
            if all(saved.get(key) == value for key, value in expected.items()):
                return (set(saved["realism"]), set(saved["influence"]),
                        set(saved["audit"]))
        except (KeyError, OSError, ValueError):
            pass

    groups: dict[tuple[str, str, str, int], list[tuple]] = defaultdict(list)
    for record in records:
        meta = record[4]
        groups[(record[2], str(meta["band"]), str(meta["pol"]),
                int(meta["depression"]))].append(record)
    realism, influence, audit = [], [], []
    for group, values in sorted(groups.items()):
        ordered = sorted(
            values,
            key=lambda record: hashlib.sha256(
                f"{seed}:{group}:{_record_name(record, root)}".encode("utf-8")
            ).hexdigest())
        audit_count, influence_count = _split_counts(
            len(ordered), influence_fraction, audit_fraction)
        audit.extend(_record_name(record, root) for record in ordered[:audit_count])
        influence.extend(_record_name(record, root)
                         for record in ordered[audit_count:audit_count + influence_count])
        realism.extend(_record_name(record, root)
                       for record in ordered[audit_count + influence_count:])
    if not realism or not influence or not audit:
        raise RuntimeError("FACT split produced an empty realism, influence, or audit set")
    payload = {**expected, "realism": sorted(realism),
               "influence": sorted(influence), "audit": sorted(audit)}
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    return set(realism), set(influence), set(audit)


def configure_records(dataset: JointROIDataset, selected: set[str], root: Path,
                      epoch_size: int = 0) -> None:
    """Restrict a JointROIDataset to one explicit FACT split."""
    dataset.records = [record for record in dataset.records
                       if _record_name(record, root) in selected]
    if not dataset.records:
        raise RuntimeError("empty FACT dataset split")
    dataset.epoch_size = epoch_size or len(dataset.records)
    dataset.random_epoch = bool(epoch_size)


def _record_acquisition(record: tuple) -> tuple[str, str, int, int]:
    meta = record[4]
    return (str(meta["band"]).upper(), str(meta["pol"]).upper(),
            int(meta["depression"]), ((int(meta["azimuth"]) + 15) % 360) // 30)


def _condition_distance(first: tuple, second: tuple) -> int:
    """A discrete distance used only to choose an identity query condition."""
    first_c = _record_acquisition(first)
    second_c = _record_acquisition(second)
    azimuth = min((first_c[3] - second_c[3]) % 12,
                  (second_c[3] - first_c[3]) % 12)
    return (4 * int(first_c[0] != second_c[0])
            + 4 * int(first_c[1] != second_c[1])
            + 2 * int(first_c[2] != second_c[2]) + azimuth)


class FactorizedInfluenceEpisodeDataset(Dataset):
    """Class-repeated, cross-condition FACT episodes from ``R_influence``.

    Each episode selects several vehicle classes and several independent SAR
    support conditions for each class.  The query is real SAR from the same
    class but a maximally different available acquisition condition.  The RGB
    view is sampled independently by class, so it is never paired to either
    SAR image.
    """

    def __init__(self, base: JointROIDataset, episodes_per_epoch: int,
                 classes_per_episode: int = 8, conditions_per_class: int = 4,
                 seed: int = 20260902) -> None:
        if episodes_per_epoch <= 0:
            raise ValueError("episodes_per_epoch must be positive")
        if classes_per_episode <= 0 or conditions_per_class < 2:
            raise ValueError("need positive classes and at least two conditions per class")
        self.base = base
        self.episodes_per_epoch = episodes_per_epoch
        self.classes_per_episode = classes_per_episode
        self.conditions_per_class = conditions_per_class
        self.seed = seed
        self.epoch = 0
        self.by_class: dict[str, list[tuple]] = defaultdict(list)
        for record in base.records:
            self.by_class[record[2]].append(record)
        self.classes = [name for name, records in sorted(self.by_class.items())
                        if len(records) >= conditions_per_class
                        and len({_record_acquisition(record) for record in records}) >= 2]
        if len(self.classes) < classes_per_episode:
            raise RuntimeError("not enough classes with cross-condition FACT records")

    @property
    def episode_size(self) -> int:
        return self.classes_per_episode * self.conditions_per_class

    def __len__(self) -> int:
        return self.episodes_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rng(self, index: int) -> random.Random:
        return random.Random(self.seed + self.epoch * 1_000_003 + int(index) * 97)

    def _support_records(self, class_name: str, rng: random.Random) -> list[tuple]:
        grouped: dict[tuple[str, str, int, int], list[tuple]] = defaultdict(list)
        for record in self.by_class[class_name]:
            grouped[_record_acquisition(record)].append(record)
        keys = list(grouped)
        rng.shuffle(keys)
        selected: list[tuple] = []
        for key in keys[:self.conditions_per_class]:
            selected.append(rng.choice(grouped[key]))
        all_records = self.by_class[class_name]
        while len(selected) < self.conditions_per_class:
            selected.append(rng.choice(all_records))
        return selected

    def _identity_query(self, support: tuple, rng: random.Random) -> tuple:
        candidates = [record for record in self.by_class[support[2]]
                      if Path(record[0]) != Path(support[0])]
        if not candidates:
            raise RuntimeError(f"FACT class has no independent query: {support[2]}")
        distances = [_condition_distance(support, record) for record in candidates]
        best = max(distances)
        if best <= 0:
            raise RuntimeError(f"FACT class has no cross-condition query: {support[2]}")
        return rng.choice([record for record, distance in zip(candidates, distances)
                           if distance == best])

    def _roi(self, record: tuple) -> torch.Tensor:
        with Image.open(record[0]) as image:
            return image_tensor(image, self.base.roi_size, False)

    def _rgb(self, class_name: str, rng: random.Random) -> torch.Tensor:
        angle = rng.choice(self.base.class_rgb_angles[class_name])
        # Influence episodes are deterministic apart from explicit generator
        # noise; source-view augmentation belongs to the main RGB pretraining.
        return self.base._rgb_base(self.base.rgb_paths[class_name, angle]).clone()

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = self._rng(index)
        classes = rng.sample(self.classes, self.classes_per_episode)
        payload: dict[str, list[torch.Tensor]] = defaultdict(list)
        for class_name in classes:
            for support in self._support_records(class_name, rng):
                query = self._identity_query(support, rng)
                support_meta = support[4]
                query_meta = query[4]
                payload["rgb"].append(self._rgb(class_name, rng))
                payload["real_support"].append(self._roi(support))
                payload["real_identity_query"].append(self._roi(query))
                payload["class_id"].append(torch.tensor(
                    self.base.class_to_id[class_name], dtype=torch.long))
                payload["support_meta"].append(metadata_vector(support_meta, support[3]))
                payload["support_depression"].append(torch.tensor(
                    int(support_meta["depression"]), dtype=torch.long))
                payload["support_azimuth"].append(torch.tensor(
                    int(support_meta["azimuth"]), dtype=torch.long))
                payload["query_depression"].append(torch.tensor(
                    int(query_meta["depression"]), dtype=torch.long))
                payload["query_azimuth"].append(torch.tensor(
                    int(query_meta["azimuth"]), dtype=torch.long))
        return {name: torch.stack(values) for name, values in payload.items()}


class RandomTaskProbe64(nn.Module):
    """A frozen randomly initialized SAR learner used only for FACT updates.

    Its convolutional representation and task heads are deterministic from a
    private seed and never optimized.  Sampling several seeds approximates a
    distribution of downstream learners instead of reusing a native SAR
    classifier decision boundary.
    """

    def __init__(self, classes: int = 40, feature_dim: int = 128,
                 base: int = 32, seed: int = 0) -> None:
        super().__init__()
        if feature_dim <= 0 or base <= 0:
            raise ValueError("feature_dim and base must be positive")
        # Probe construction occurs on CPU before the bank is moved to its
        # device.  Use a private generator and restore the default CPU state:
        # calling ``torch.manual_seed`` here would also perturb CUDA streams
        # used later for GAN noise in every visible device.
        default_cpu_rng = torch.random.get_rng_state()
        private_rng = torch.Generator(device="cpu").manual_seed(int(seed))
        try:
            self.trunk = nn.Sequential(
                nn.Conv2d(1, base, 3, 1, 1, bias=False),
                nn.GroupNorm(8, base), nn.SiLU(),
                nn.Conv2d(base, base * 2, 4, 2, 1, bias=False),
                nn.GroupNorm(8, base * 2), nn.SiLU(),
                nn.Conv2d(base * 2, feature_dim, 4, 2, 1, bias=False),
                nn.GroupNorm(8, feature_dim), nn.SiLU(),
                nn.Conv2d(feature_dim, feature_dim, 4, 2, 1, bias=False),
                nn.GroupNorm(8, feature_dim), nn.SiLU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.LayerNorm(feature_dim),
            )
            self.register_buffer("class_weight", torch.empty(classes, feature_dim))
            self.register_buffer("class_bias", torch.empty(classes))
            self.register_buffer("band_weight", torch.empty(2, feature_dim))
            self.register_buffer("band_bias", torch.empty(2))
            self.register_buffer("polarization_weight", torch.empty(4, feature_dim))
            self.register_buffer("polarization_bias", torch.empty(4))
            self.register_buffer("depression_weight", torch.empty(4, feature_dim))
            self.register_buffer("depression_bias", torch.empty(4))
            self.register_buffer("azimuth_weight", torch.empty(12, feature_dim))
            self.register_buffer("azimuth_bias", torch.empty(12))
            for module in self.modules():
                if isinstance(module, nn.Conv2d):
                    fan_in = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
                    fan_in //= module.groups
                    with torch.no_grad():
                        module.weight.normal_(0.0, math.sqrt(2.0 / fan_in),
                                              generator=private_rng)
            for weight in (self.class_weight, self.band_weight,
                           self.polarization_weight, self.depression_weight,
                           self.azimuth_weight):
                with torch.no_grad():
                    weight.normal_(0.0, 1.0 / math.sqrt(feature_dim),
                                   generator=private_rng)
            for bias in (self.class_bias, self.band_bias, self.polarization_bias,
                         self.depression_bias, self.azimuth_bias):
                nn.init.zeros_(bias)
        finally:
            torch.random.set_rng_state(default_cpu_rng)
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.trunk(image)

    def acquisition_heads(self) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        return ((self.band_weight, self.band_bias),
                (self.polarization_weight, self.polarization_bias),
                (self.depression_weight, self.depression_bias),
                (self.azimuth_weight, self.azimuth_bias))


class RandomTaskProbeBank(nn.Module):
    """A fixed bank of independently seeded FACT learners."""

    def __init__(self, count: int = 4, classes: int = 40,
                 feature_dim: int = 128, seed: int = 20260902) -> None:
        super().__init__()
        if count <= 0:
            raise ValueError("probe count must be positive")
        self.probes = nn.ModuleList(
            RandomTaskProbe64(classes, feature_dim, seed=seed + 10_007 * index)
            for index in range(count))

    def select(self, event: int) -> RandomTaskProbe64:
        return self.probes[int(event) % len(self.probes)]


def _per_sample_linear_ce_gradient(features: torch.Tensor, weight: torch.Tensor,
                                   bias: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Exact per-sample CE gradient for a frozen linear task head.

    Representing updates analytically avoids calling ``autograd.grad`` on a
    mutable teacher head.  When ``features`` come from generated SAR, this
    expression remains differentiable all the way to the generator.
    """
    logits = F.linear(features, weight, bias)
    residual = logits.softmax(1) - F.one_hot(target.long(), logits.shape[1]).to(logits.dtype)
    gradient_weight = residual.unsqueeze(2) * features.unsqueeze(1)
    return torch.cat((gradient_weight.flatten(1), residual), dim=1)


def _mean_by_group(values: torch.Tensor, groups: torch.Tensor
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return group means, their row-to-group inverse, and ordered group ids."""
    group_ids, inverse = torch.unique(groups.long(), sorted=True, return_inverse=True)
    means = values.new_zeros(len(group_ids), values.shape[1])
    means.index_add_(0, inverse, values)
    counts = torch.bincount(inverse, minlength=len(group_ids)).to(values.dtype)
    means = means / counts[:, None].clamp_min(1.0)
    return means, inverse, group_ids


def _cosine_distance(first: torch.Tensor, second: torch.Tensor,
                     eps: float = 1e-6) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(first, second, dim=1, eps=eps).mean()


def factorized_training_influence_loss(
        probe: RandomTaskProbe64,
        fake_support: torch.Tensor,
        real_support: torch.Tensor,
        real_identity_query: torch.Tensor,
        class_id: torch.Tensor,
        support_meta: torch.Tensor,
        support_depression: torch.Tensor,
        support_azimuth: torch.Tensor,
        eps: float = 1e-6) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute FACT's single factorized training-influence objective.

    ``fake_support`` is differentiable; both real tensors and all probe
    parameters are detached.  Identity updates are compared through a real
    same-class, different-condition query.  Acquisition updates are compared
    after subtracting each class's average update, isolating the acquisition
    residual rather than rewarding a class-specific classifier shortcut.
    """
    tensors = (fake_support, real_support, real_identity_query)
    if any(value.ndim != 4 or value.shape[1] != 1 for value in tensors):
        raise ValueError("FACT images must be [B,1,H,W]")
    batch = len(fake_support)
    if batch < 2 or len(real_support) != batch or len(real_identity_query) != batch:
        raise ValueError("FACT support/query batches must have the same size >= 2")
    if len(class_id) != batch:
        raise ValueError("FACT class labels do not match support batch")

    fake_features = probe(fake_support.float().clamp(0, 1))
    with torch.no_grad():
        real_features = probe(real_support.float().clamp(0, 1))
        query_features = probe(real_identity_query.float().clamp(0, 1))

    fake_class = _per_sample_linear_ce_gradient(
        fake_features, probe.class_weight, probe.class_bias, class_id)
    with torch.no_grad():
        real_class = _per_sample_linear_ce_gradient(
            real_features, probe.class_weight, probe.class_bias, class_id)
        query_class = _per_sample_linear_ce_gradient(
            query_features, probe.class_weight, probe.class_bias, class_id)

    fake_identity, _, class_groups = _mean_by_group(fake_class, class_id)
    real_identity, _, _ = _mean_by_group(real_class, class_id)
    query_identity, _, _ = _mean_by_group(query_class, class_id)
    identity_direction = _cosine_distance(fake_identity, real_identity, eps)
    fake_query_effect = F.cosine_similarity(
        fake_identity, query_identity, dim=1, eps=eps)
    real_query_effect = F.cosine_similarity(
        real_identity, query_identity, dim=1, eps=eps)
    identity_query = F.smooth_l1_loss(fake_query_effect, real_query_effect)
    identity_loss = .5 * (identity_direction + identity_query)

    targets = acquisition_targets(support_meta, support_depression, support_azimuth)
    fake_acquisition = torch.cat([
        _per_sample_linear_ce_gradient(fake_features, weight, bias, target)
        for (weight, bias), target in zip(probe.acquisition_heads(), targets)
    ], dim=1)
    with torch.no_grad():
        real_acquisition = torch.cat([
            _per_sample_linear_ce_gradient(real_features, weight, bias, target)
            for (weight, bias), target in zip(probe.acquisition_heads(), targets)
        ], dim=1)

    fake_class_mean, fake_class_inverse, _ = _mean_by_group(fake_acquisition, class_id)
    real_class_mean, real_class_inverse, _ = _mean_by_group(real_acquisition, class_id)
    fake_residual = fake_acquisition - fake_class_mean[fake_class_inverse]
    real_residual = real_acquisition - real_class_mean[real_class_inverse]
    condition_groups = acquisition_code(support_meta, support_depression, support_azimuth)
    fake_condition, _, condition_ids = _mean_by_group(fake_residual, condition_groups)
    real_condition, _, _ = _mean_by_group(real_residual, condition_groups)
    acquisition_direction = _cosine_distance(fake_condition, real_condition, eps)
    acquisition_magnitude = F.smooth_l1_loss(
        fake_condition.norm(dim=1), real_condition.norm(dim=1))
    acquisition_loss = .5 * (acquisition_direction + acquisition_magnitude)

    total = .5 * (identity_loss + acquisition_loss)
    diagnostics = {
        "fact_identity": identity_loss.detach(),
        "fact_identity_direction": identity_direction.detach(),
        "fact_identity_query": identity_query.detach(),
        "fact_acquisition": acquisition_loss.detach(),
        "fact_acquisition_direction": acquisition_direction.detach(),
        "fact_acquisition_magnitude": acquisition_magnitude.detach(),
        "fact_class_groups": fake_support.new_tensor(float(len(class_groups))),
        "fact_condition_groups": fake_support.new_tensor(float(len(condition_ids))),
        "fact_query_effect_fake": fake_query_effect.detach().mean(),
        "fact_query_effect_real": real_query_effect.detach().mean(),
    }
    return total, diagnostics
