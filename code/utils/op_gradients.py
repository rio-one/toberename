"""Observation-protected SDS gradients, without changing the optimizer."""
import torch
import torch.distributed as dist


def _average_gradients(gradients, parameters):
    """DDP no_sync is used by the caller; average before projecting."""
    if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        return gradients
    present = torch.tensor([g is not None for g in gradients],
                           device=parameters[0].device, dtype=torch.int32)
    dist.all_reduce(present)
    result = []
    for count, gradient, parameter in zip(present.tolist(), gradients, parameters):
        if count == 0:
            result.append(None)
            continue
        gradient = torch.zeros_like(parameter) if gradient is None else gradient.contiguous()
        dist.all_reduce(gradient)
        result.append(gradient / dist.get_world_size())
    return result


def observation_protected_backward(reconstruction_loss, prior_loss, parameters,
                                   parameter_groups, strength=1.0, eps=1e-12):
    """Assign merged .grad, projecting within named parameter groups.

    The prior must already include its SDS weight and visibility weighting.
    Only parameters reached by the prior are eligible for projection: this
    must not introduce prior updates into unrelated rendering/density fields.
    Frozen and globally unused parameters retain grad=None. The optimizer is
    stepped by the caller. With Adam this is gradient control, not a guarantee
    that the actual finite optimizer step decreases reconstruction loss.
    """
    if not 0.0 <= strength <= 1.0 or eps <= 0:
        raise ValueError('OP projection requires strength in [0, 1] and eps > 0')
    parameters = list(dict.fromkeys(p for p in parameters if p.requires_grad))
    if not parameters:
        raise ValueError('No trainable optimizer parameters')
    rec = list(torch.autograd.grad(reconstruction_loss, parameters,
                                  retain_graph=True, allow_unused=True))
    prior = (list(torch.autograd.grad(prior_loss, parameters, allow_unused=True))
             if prior_loss.requires_grad else [None] * len(parameters))
    rec = _average_gradients(rec, parameters)
    prior = _average_gradients(prior, parameters)
    lookup = {id(p): i for i, p in enumerate(parameters)}
    groups, used = [], set()
    for name, members in parameter_groups.items():
        indices = [lookup[id(p)] for p in members if id(p) in lookup]
        if used.intersection(indices):
            raise ValueError('Overlapping OP parameter groups')
        used.update(indices)
        groups.append((name, indices))
    groups.append(('other', [i for i in range(len(parameters)) if i not in used]))
    zero = reconstruction_loss.detach().new_zeros((), dtype=torch.float32)
    original_sq, projected_sq = zero.clone(), zero.clone()
    cosines, conflicts = [], []
    for _, indices in groups:
        active = [i for i in indices if prior[i] is not None]
        if not active:
            continue
        dot, rec_sq, prior_sq = zero.clone(), zero.clone(), zero.clone()
        for i in active:
            gd = prior[i].float()
            prior_sq = prior_sq + gd.square().sum()
            if rec[i] is not None:
                gr = rec[i].float()
                dot = dot + (gr * gd).sum()
                rec_sq = rec_sq + gr.square().sum()
        valid = (rec_sq > eps) & (prior_sq > eps)
        coefficient = torch.where(valid, strength * dot.clamp(max=0) / rec_sq.clamp_min(eps), zero)
        if bool(valid):
            cosines.append(dot / (rec_sq.sqrt() * prior_sq.sqrt()))
            conflicts.append((dot < 0).float())
        original_sq = original_sq + prior_sq
        for i in active:
            if rec[i] is not None:
                prior[i] = prior[i] - coefficient.to(prior[i].dtype) * rec[i]
            projected_sq = projected_sq + prior[i].float().square().sum()
    for parameter, gr, gd in zip(parameters, rec, prior):
        parameter.grad = gd if gr is None else gr if gd is None else gr + gd
    return {
        'op_conflict_rate': torch.stack(conflicts).mean().detach() if conflicts else zero,
        'op_gradient_cosine': torch.stack(cosines).mean().detach() if cosines else zero,
        'op_prior_retention': torch.where(original_sq > eps,
            (projected_sq / original_sq.clamp_min(eps)).sqrt(), torch.ones_like(zero)).detach(),
    }
