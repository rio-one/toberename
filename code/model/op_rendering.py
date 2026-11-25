"""Exclusive instance attribution using the existing scene hit weights."""
import torch
import torch.nn.functional as F


def sdf_ownership(sdf_raw, beta):
    """Per-sample instance probabilities; no semantic head or extra field query.

    beta sets the spatial transition scale. Detach only this temperature so
    instance assignment cannot sharpen itself by changing density thickness;
    beta still receives gradients through the original scene hit weights.
    """
    if sdf_raw.ndim != 2 or sdf_raw.shape[-1] < 1:
        raise ValueError("Expected SDF channels with shape [points, instances]")
    temperature = torch.as_tensor(beta, device=sdf_raw.device,
                                  dtype=sdf_raw.dtype).detach().clamp_min(1e-6)
    return torch.softmax(-sdf_raw / temperature, dim=-1)


def exclusive_object_opacity(weights, sdf_raw, beta, object_index=None):
    """Sum shared hit weights times pointwise ownership, without ray renormalization."""
    if weights.ndim != 2 or sdf_raw.ndim != 2 or sdf_raw.shape[0] != weights.numel():
        raise ValueError("Scene weights and SDF sample counts do not match")
    classes = sdf_raw.shape[-1]
    if object_index is not None:
        if not 0 <= object_index < classes:
            raise ValueError("Object index is outside the SDF channel range")
        selected = F.one_hot(torch.as_tensor(object_index, device=weights.device),
                             num_classes=classes).to(weights.dtype)
        return weights.sum(dim=-1, keepdim=True) * selected
    ownership = sdf_ownership(sdf_raw, beta).reshape(*weights.shape, classes)
    return (weights.unsqueeze(-1) * ownership).sum(dim=1)
