"""Scene/rotating-instance references from current training observations."""
import torch


def observation_references(per_ray, labels, reliability, valid, step,
                           instance_enabled=True, interval=4, min_pixels=4):
    per_ray, labels = per_ray.reshape(-1), labels.reshape(-1).long()
    reliability = reliability.reshape(-1).to(per_ray).detach().clamp(0, 1)
    valid = valid.reshape(-1).to(device=per_ray.device, dtype=torch.bool)
    valid = valid & torch.isfinite(per_ray.detach())
    weight = reliability * valid
    def reference(mask):
        w = weight * mask
        loss = (torch.where(valid, per_ray, torch.zeros_like(per_ray)) * w).sum() / w.sum().clamp_min(1e-8)
        confidence = w.sum() / (valid & mask).sum().clamp_min(1)
        return loss, confidence.detach()
    scene = reference(torch.ones_like(valid))
    references = [scene]
    selected = -1
    if instance_enabled and step % interval == 0:
        candidates = [int(i) for i in labels[valid & (weight > 0)].unique().tolist()
                      if i > 0 and int(((labels == i) & valid & (weight > 0)).sum()) >= min_pixels]
        if candidates:
            # All IDs share one rule, no scene/object-specific exception table.
            selected = candidates[(step // interval) % len(candidates)]
            references.append(reference(labels == selected))
    return references, selected
