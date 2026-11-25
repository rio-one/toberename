"""Locally conditioned residual SDF; shares the existing hash and visibility grids."""
from contextlib import contextmanager
import math
import weakref

import torch
from torch import nn


def sample_frozen_visibility(points, visibility):
    """Trilinear VisGrid lookup with spatial double backward and zero padding.

    grid_sample's 3-D backward lacks the double backward needed by Eikonal.
    Gather detached corner values instead: no grid copy or grid parameter graph.
    VisGrid reverses x/z for grid_sample, so its stored D/H/W axes are x/y/z.
    """
    grid = visibility.visgrid.detach()
    if grid.ndim != 5 or grid.shape[:2] != (1, 1) or len(set(grid.shape[2:])) != 1:
        raise ValueError("Residual SDF requires the existing cubic VisGrid")
    resolution = grid.shape[-1]
    coord = (points / visibility.divide_factor + 1) * ((resolution - 1) / 2)
    lower = coord.detach().floor().long()
    fraction = coord - lower.to(coord.dtype)
    values = points.new_zeros((len(points), 1))
    flat = grid.reshape(-1)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                ix, iy, iz = lower[:, 0] + dx, lower[:, 1] + dy, lower[:, 2] + dz
                inside = ((ix >= 0) & (ix < resolution) & (iy >= 0) &
                          (iy < resolution) & (iz >= 0) & (iz < resolution))
                index = ((ix.clamp(0, resolution-1) * resolution +
                          iy.clamp(0, resolution-1)) * resolution +
                         iz.clamp(0, resolution-1))
                weight = ((fraction[:, 0] if dx else 1-fraction[:, 0]) *
                          (fraction[:, 1] if dy else 1-fraction[:, 1]) *
                          (fraction[:, 2] if dz else 1-fraction[:, 2]))
                corner = flat[index] * inside.to(points.dtype)
                values = values + (weight * corner).unsqueeze(-1)
    return values


class ResidualSDF(nn.Module):
    """Small decoder with optional local evidence, shared across all instance IDs.

    The legacy coordinate-only layout remains available for v4 configurations.
    Local conditioning reuses base SDF/hash outputs from the same forward pass;
    its visibility input is the very same gate used to compose the final field.
    """

    def __init__(self, outputs, width=32, bands=3, max_residual=0.2,
                 hidden_visibility=0.1, visible_visibility=0.5,
                 coordinate_scale=1.0, chunk_size=8192,
                 conditioned=False, feature_dim=0):
        super().__init__()
        if (outputs < 1 or width < 1 or bands < 0 or max_residual <= 0 or
                coordinate_scale <= 0 or chunk_size < 1 or feature_dim < 0 or
                (not conditioned and feature_dim != 0) or
                not 0 <= hidden_visibility < visible_visibility <= 1):
            raise ValueError("Invalid residual SDF configuration")
        self.outputs, self.bands = outputs, bands
        self.conditioned, self.feature_dim = bool(conditioned), int(feature_dim)
        self.max_residual = float(max_residual)
        self.hidden_visibility = float(hidden_visibility)
        self.visible_visibility = float(visible_visibility)
        self.coordinate_scale, self.chunk_size = float(coordinate_scale), int(chunk_size)
        input_dim = 3 + 6*bands
        if self.conditioned:
            input_dim += 1 + outputs + self.feature_dim
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, width), nn.Softplus(beta=10),
            nn.Linear(width, outputs))
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)
        self.register_buffer("active_flag", torch.tensor(False))
        self.active = False
        self._visibility = None  # weak reference: never register/copy the large grid

    def bind_visibility(self, visibility):
        self._visibility = weakref.ref(visibility)

    def activate(self):
        if self._visibility is None or self._visibility() is None:
            raise RuntimeError("Bind the checkpoint's VisGrid before activating residual SDF")
        if not self.active:
            self.active_flag.fill_(True)
            self.active = True

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self.active = bool(self.active_flag.item())

    def gate(self, points):
        visibility = None if self._visibility is None else self._visibility()
        if visibility is None:
            raise RuntimeError("Residual SDF inference/export needs its trained VisGrid")
        observed = sample_frozen_visibility(points, visibility)
        t = ((observed-self.hidden_visibility) /
             (self.visible_visibility-self.hidden_visibility)).clamp(0, 1)
        return 1 - t.square() * (3 - 2*t)

    def _correction(self, points, base_sdf=None, features=None):
        coords = points / self.coordinate_scale
        encoded = [coords]
        for band in range(self.bands):
            phase = coords * (math.pi * 2**band)
            encoded.extend((phase.sin(), phase.cos()))
        gate = self.gate(points)
        if self.conditioned:
            # Keep spatial derivatives through every condition, including the gate.
            # Parameter isolation is applied at the prior forward, not via detach.
            encoded.extend((gate, (base_sdf / self.max_residual).tanh()))
            if self.feature_dim:
                encoded.append(features)
        residual = self.max_residual * self.decoder(torch.cat(encoded, -1)).tanh()
        return gate * residual

    def forward(self, points, base_sdf=None, features=None):
        if self.conditioned:
            if base_sdf is None or base_sdf.shape != (len(points), self.outputs):
                raise ValueError("Conditioned residual SDF needs matching base SDF channels")
            if self.feature_dim and (features is None or
                                     features.shape != (len(points), self.feature_dim)):
                raise ValueError("Conditioned residual SDF needs its shared coarse hash features")
        if len(points) == 0:
            return points.new_zeros((0, self.outputs))
        parts = []
        for start in range(0, len(points), self.chunk_size):
            stop = start + self.chunk_size
            parts.append(self._correction(
                points[start:stop],
                None if base_sdf is None else base_sdf[start:stop],
                None if features is None else features[start:stop]))
        return torch.cat(parts, 0)


@contextmanager
def isolated_completion_prior(implicit, *other_modules):
    """Freeze parameter leaves for the prior forward, retaining x derivatives.

    Observation graphs are built before this context and keep their own parameter
    connections. Restoring flags before backward leaves those graphs trainable.
    """
    completion = getattr(implicit, "completion", None)
    if completion is None or not completion.active:
        yield
        return
    allowed = {id(p) for p in completion.parameters()}
    frozen = []
    seen = set()
    for module in (implicit,) + other_modules:
        for parameter in module.parameters():
            if id(parameter) not in seen and id(parameter) not in allowed:
                seen.add(id(parameter))
                if parameter.requires_grad:
                    frozen.append(parameter)
    try:
        for parameter in frozen:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter in frozen:
            parameter.requires_grad_(True)
