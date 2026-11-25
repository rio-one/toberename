"""Bounded training-view surface checks; no GT meshes, novel views or extra model."""
import math
import torch
import torch.nn.functional as F
from utils import rend_util


def backproject(uv, depth, pose, intrinsics):
    pixel = torch.cat((uv, torch.ones_like(uv[:, :1])), -1)
    camera = (pixel @ torch.linalg.inv(intrinsics[:3, :3]).T) * depth.reshape(-1, 1)
    return camera @ pose[:3, :3].T + pose[:3, 3]


def project(points, pose, intrinsics):
    camera = (points - pose[:3, 3]) @ pose[:3, :3]
    homogeneous = camera @ intrinsics[:3, :3].T
    uv = homogeneous[:, :2] / homogeneous[:, 2:].clamp_min(1e-8)
    return uv, camera[:, 2]


def pair_validity(projected_uv, projected_z, target_depth, target_normal, source_normal,
                  source_label, target_label, source_valid, target_valid, height, width,
                  depth_tolerance=0.05, normal_cos_min=0.5):
    """Detached, conservative overlap/occlusion gate in shared scene coordinates."""
    uv, z = projected_uv.detach(), projected_z.detach()
    depth = target_depth.detach().reshape(-1)
    relative = (z - depth).abs() / depth.abs().clamp_min(0.05)
    cosine = F.cosine_similarity(source_normal.detach(), target_normal.detach(), dim=-1)
    inside = ((uv[:, 0] >= 0) & (uv[:, 0] <= width-1)
              & (uv[:, 1] >= 0) & (uv[:, 1] <= height-1) & (z > 0))
    finite = (torch.isfinite(uv).all(-1) & torch.isfinite(z) & torch.isfinite(depth)
              & torch.isfinite(cosine) & (depth > 0))
    valid = (inside & finite & source_valid.reshape(-1).bool() & target_valid.reshape(-1).bool()
             & (source_label.reshape(-1) == target_label.reshape(-1))
             & (relative <= depth_tolerance) & (cosine >= normal_cos_min))
    confidence = torch.where(valid, torch.exp(-relative/depth_tolerance) * cosine.clamp(0, 1),
                             torch.zeros_like(relative))
    return valid, confidence.detach()


class TrainingViewConsistency:
    def __init__(self, dataset, max_points=32, interval=8, warmup=5000, ramp=2000,
                 depth_tolerance=0.05, normal_cos_min=0.5, depth_weight=0.05,
                 normal_weight=0.01, reference_floor=0.5, ttl=2000):
        if (max_points < 1 or interval < 1 or warmup < 0 or ramp < 1 or ttl < 1
                or depth_tolerance <= 0 or not -1 <= normal_cos_min <= 1
                or not 0 <= reference_floor <= 1 or depth_weight < 0 or normal_weight < 0):
            raise ValueError('Invalid light OP multiview configuration')
        self.dataset = dataset
        self.height, self.width = dataset.img_res
        self.max_points, self.interval, self.warmup, self.ramp = max_points, interval, warmup, ramp
        self.depth_tolerance, self.normal_cos_min = depth_tolerance, normal_cos_min
        self.depth_weight, self.normal_weight = depth_weight, normal_weight
        self.floor, self.ttl = reference_floor, ttl
        # Dataset contains training images. Respect an optional training-view subset.
        self.views = (list(range(dataset.n_images)) if dataset.num_views < 0 else
                      [25, 22, 28, 40, 44, 48, 0, 8, 13][:dataset.num_views])
        if any(view >= dataset.n_images for view in self.views):
            raise ValueError('Training-view subset is out of range')
        self.neighbors = {}
        for source in self.views:
            pose = dataset.pose_all[source]
            others = [v for v in self.views if v != source]
            self.neighbors[source] = sorted(others, key=lambda v: float(
                (pose[:3, 3]-dataset.pose_all[v][:3, 3]).square().sum()
                + 0.1 * (1 - torch.dot(pose[:3, 2], dataset.pose_all[v][:3, 2]))))[:3]
        # Sparse CPU evidence; no full-image GPU cache and no checkpoint files.
        self.evidence = {}
        self.tables = {}

    def _keys(self, uv):
        x = uv.detach().cpu().round().long()
        x[:, 0].clamp_(0, self.width-1);x[:, 1].clamp_(0, self.height-1)
        return x[:, 1] * self.width + x[:, 0]

    def reliability(self, view, uv, step):
        keys = self._keys(uv)
        values = torch.full((len(keys),), self.floor)
        evidence = self.evidence.get(view, {})
        if evidence:
            if view not in self.tables:
                ordered = sorted(evidence)
                self.tables[view] = (torch.tensor(ordered),
                    torch.tensor([evidence[k][0] for k in ordered]),
                    torch.tensor([evidence[k][1] for k in ordered]))
            known, confidence, updated = self.tables[view]
            position = torch.searchsorted(known, keys).clamp_max(len(known)-1)
            matched = known[position] == keys
            freshness = torch.exp(-(step-updated[position]).clamp_min(0).float()/self.ttl)
            values[matched] = (self.floor+(confidence[position]-self.floor)*freshness)[matched]
        return values.to(uv.device)

    def _update(self, view, uv, confidence, step):
        self.tables.pop(view, None)
        evidence = self.evidence.setdefault(view, {})
        # A checked invalid pair is weak evidence, not permanent rejection.
        for key, value in zip(self._keys(uv).tolist(), confidence.detach().cpu().tolist()):
            evidence[key] = (max(0.1, min(1.0, value)), step)
        # Bound history even for long/high-resolution runs.
        if len(evidence) > 8192:
            recent = sorted(evidence, key=lambda key: evidence[key][1], reverse=True)[:4096]
            self.evidence[view] = {key: evidence[key] for key in recent}

    def _target(self, view, step):
        neighbors = self.neighbors.get(view, [])
        return neighbors[(step // self.interval) % len(neighbors)] if neighbors else None

    @torch.no_grad()
    def _query_surface(self, model, uv, pose, intrinsics):
        """Only bounded SDF/depth queries; no RGB, eikonal, background or SDS forward."""
        ray, origin = rend_util.get_camera_params(uv[None], pose[None], intrinsics[None])
        ray = ray[0];origin = origin[0].expand_as(ray)
        z, _ = model.ray_sampler.get_z_vals(ray, origin, model)
        points = origin[:, None] + z[..., None] * ray[:, None]
        sdf = model.implicit_network.get_sdf_vals(points.reshape(-1, 3))
        weights, _, _ = model.volume_rendering(z, sdf)
        distance = (weights*z).sum(-1) / weights.sum(-1).clamp_min(1e-8)
        surface = origin + distance[:, None]*ray
        _, depth = project(surface, pose, intrinsics)
        raw = sdf.reshape(z.shape)
        valid = (raw.min(-1).values < 0) & (raw.max(-1).values > 0) & (weights.sum(-1) > 0.9)
        with torch.enable_grad():
            position = surface.detach().requires_grad_(True)
            field = model.implicit_network.get_sdf_vals(position)
            normal = torch.autograd.grad(field.sum(), position, create_graph=False)[0]
        return depth.detach(), F.normalize(normal.detach(), dim=-1), valid.detach()

    def check(self, model, view, uv, points, normals, labels, source_valid, step,
              query_surface=None, geometry=True):
        """Geometry loss uses a detached target surface; texture only updates reliability."""
        zero = points.sum()*0
        stats = {'op_mv_pairs': zero.detach(), 'op_mv_checked': zero.detach(),
                 'op_mv_loss': zero.detach()}
        target = self._target(view, step)
        start = self.warmup if geometry else 0
        if target is None or step < start or step % self.interval:
            return zero, stats
        selected = torch.nonzero(source_valid.reshape(-1).detach(), as_tuple=False).flatten()
        if len(selected) == 0:
            return zero, stats
        selected = selected[torch.randperm(len(selected), device=selected.device)[:self.max_points]]
        device = points.device
        pose = self.dataset.pose_all[target].to(device)
        intrinsics = self.dataset.intrinsics_all[target].to(device)
        projected, z = project(points[selected], pose, intrinsics)
        detached_uv = projected.detach()
        in_bounds = (torch.isfinite(detached_uv).all(-1) & torch.isfinite(z.detach())
                     & (detached_uv[:, 0] >= 0) & (detached_uv[:, 0] <= self.width-1)
                     & (detached_uv[:, 1] >= 0) & (detached_uv[:, 1] <= self.height-1) & (z.detach() > 0))
        # Never send invalid/behind-camera rays to the surface sampler.
        candidates = torch.nonzero(in_bounds, as_tuple=False).flatten()
        confidence = points.new_zeros(len(selected))
        if len(candidates) == 0:
            self._update(view, uv[selected], confidence, step)
            stats['op_mv_checked'] = zero.new_tensor(len(selected)).detach()
            return zero, stats
        pixel = detached_uv[candidates].round().long()
        target_ids = pixel[:, 1] * self.width + pixel[:, 0]
        target_labels = self.dataset.semantic_images[target][target_ids.cpu()].reshape(-1).to(device).long()
        target_mask = self.dataset.mask_images[target][target_ids.cpu()].reshape(-1).to(device) > 0.5
        source_ids = selected[candidates]
        same_label = (target_labels == labels.reshape(-1)[source_ids]) & target_mask
        # Query only same-instance candidates, bounded by max_points.
        candidates = candidates[same_label]
        source_ids = selected[candidates]
        if len(candidates) == 0:
            self._update(view, uv[selected], confidence, step)
            stats['op_mv_checked'] = zero.new_tensor(len(selected)).detach()
            return zero, stats
        pixel = detached_uv[candidates].round().long()
        ids = pixel[:, 1]*self.width+pixel[:, 0]
        target_labels = self.dataset.semantic_images[target][ids.cpu()].reshape(-1).to(device).long()
        query = query_surface or (lambda u, p, k: self._query_surface(model, u, p, k))
        depth, target_normal, target_valid = query(detached_uv[candidates], pose, intrinsics)
        valid, pair_confidence = pair_validity(projected[candidates], z[candidates], depth, target_normal,
            normals[source_ids], labels.reshape(-1)[source_ids], target_labels,
            torch.ones_like(target_valid), target_valid, self.height, self.width,
            self.depth_tolerance, self.normal_cos_min)
        confidence[candidates] = pair_confidence
        self._update(view, uv[selected], confidence, step)
        if bool(valid.any()) and geometry:
            relative = (z[candidates]-depth.detach()) / depth.detach().abs().clamp_min(0.05)
            depth_term = F.smooth_l1_loss(relative[valid], torch.zeros_like(relative[valid]), beta=self.depth_tolerance)
            normal_term = (1-F.cosine_similarity(normals[source_ids][valid], target_normal.detach()[valid], dim=-1)).mean()
            ramp = min(1.0, (step-self.warmup+1)/self.ramp)
            loss = ramp*(self.depth_weight*depth_term+self.normal_weight*normal_term)
        else:
            loss = zero
        stats = {'op_mv_pairs': zero.new_tensor(int(valid.sum())).detach(),
                 'op_mv_checked': zero.new_tensor(len(selected)).detach(), 'op_mv_loss': loss.detach()}
        return loss, stats

    def geometry(self, model, view, model_input, outputs, ground_truth, step, query_surface=None):
        uv = model_input['uv'][0]
        points = backproject(uv, outputs['depth_values'], model_input['pose'][0], model_input['intrinsics'][0])
        normals = outputs['normal_map'] @ model_input['pose'][0, :3, :3].T
        sdf = outputs['sdf'].detach()
        valid = ((sdf.min(-1).values < 0) & (sdf.max(-1).values > 0)
                 & (outputs['weights'].detach().sum(-1) > 0.9)
                 & (ground_truth['mask'].to(uv.device).reshape(-1) > 0.5))
        return self.check(model, view, uv, points, normals, ground_truth['segs'].to(uv.device).long(),
                          valid, step, query_surface=query_surface)

    def texture(self, model, view, uv, out, labels, step, query_surface=None):
        points = out['surface_position'][0].reshape(-1, 3).detach()
        normals = out['surface_normal_world'][0].reshape(-1, 3).detach()
        valid = out['opacity'][0].reshape(-1).detach() > 0.99
        return self.check(model, view, uv, points, normals, labels.to(uv.device).long(),
                          valid, step, query_surface=query_surface, geometry=False)
