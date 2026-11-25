"""Small observation/real-neighbor losses, independent of diffusion models."""
import torch


def masked_rgb_mse(prediction, target, mask):
    """Average over covered RGB elements; empty coverage has zero gradient."""
    weight = mask.detach().to(prediction).expand_as(prediction)
    return ((prediction - target).square() * weight).sum() / weight.sum().clamp_min(1)


def real_neighbor_depth_loss(prediction, target, mask, uv, labels, alpha=0.5):
    """Per-view scale/shift fit; residual gradients only across real same-label edges.

    Coordinates are the sampled image pixels, never a reshaped random-ray grid.
    Duplicate pixels count once in the edge term; no edges gives a zero term.
    Monocular depth remains an auxiliary prior, not a protected observation.
    """
    batch = uv.shape[0]
    prediction = prediction.reshape(batch, -1)
    target = target.reshape(batch, -1)
    mask = mask.reshape(batch, -1).bool()
    labels = labels.reshape(batch, -1)
    uv = uv.reshape(batch, -1, 2).long()
    valid = mask & torch.isfinite(prediction) & torch.isfinite(target) & (target > 0)
    pred = torch.where(valid, prediction, torch.zeros_like(prediction))
    truth = torch.where(valid, target, torch.zeros_like(target))
    weight = valid.to(pred)
    count = weight.sum(1).clamp_min(1)
    mean_p, mean_t = pred.sum(1) / count, truth.sum(1) / count
    dp, dt = (pred - mean_p[:, None]) * weight, (truth - mean_t[:, None]) * weight
    variance = dp.square().sum(1)
    scale = torch.where(variance > 1e-8, (dp * dt).sum(1) / variance.clamp_min(1e-8),
                        torch.zeros_like(variance))
    residual = (scale[:, None] * (pred - mean_p[:, None]) + mean_t[:, None] - truth) * weight
    data_loss = residual.square().sum() / (2 * weight.sum().clamp_min(1))
    edge_sum, edge_count = residual.sum() * 0, residual.new_zeros(())
    for b in range(batch):
        # A stride larger than sampled x prevents wrapping at row ends.
        stride = uv[b, :, 0].max() + 2
        keys = uv[b, :, 1] * stride + uv[b, :, 0]
        order = torch.argsort(keys)
        sorted_keys = keys[order]
        unique = torch.ones_like(sorted_keys, dtype=torch.bool)
        unique[1:] = sorted_keys[1:] != sorted_keys[:-1]
        order = order[unique]
        keys = keys[order]
        for offset in (1, stride):
            destination = torch.searchsorted(keys, keys + offset)
            safe = destination.clamp_max(len(keys) - 1)
            other = order[safe]
            keep = ((destination < len(keys)) & (keys[safe] == keys + offset)
                    & valid[b, order] & valid[b, other]
                    & (labels[b, order] == labels[b, other]))
            edge_sum = edge_sum + (residual[b, order] - residual[b, other]).abs()[keep].sum()
            edge_count = edge_count + keep.sum()
    return data_loss + alpha * edge_sum / edge_count.clamp_min(1)
