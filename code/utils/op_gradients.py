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
                                   parameter_groups, strength=1.0, eps=1e-12, reference_loss=None):
    """Assign merged .grad, projecting within named parameter groups.

    reconstruction_loss includes every non-SDS training term. reference_loss
    is the direct observation target; None preserves the previous full-loss
    reference for ablation. The prior includes SDS/visibility weighting.
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
    def gradients(loss):
        return (list(torch.autograd.grad(loss, parameters, retain_graph=True,
                                         allow_unused=True))
                if loss.requires_grad else [None] * len(parameters))
    # Keep all reconstruction/regularization gradients in the optimizer update.
    # Only the separately supplied observation gradient is a protection target.
    rec = gradients(reconstruction_loss)
    reference = rec if reference_loss is None else gradients(reference_loss)
    prior = (list(torch.autograd.grad(prior_loss, parameters, allow_unused=True))
             if prior_loss.requires_grad else [None] * len(parameters))
    rec = _average_gradients(rec, parameters)
    reference = rec if reference_loss is None else _average_gradients(reference, parameters)
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
            if reference[i] is not None:
                gr = reference[i].float()
                dot = dot + (gr * gd).sum()
                rec_sq = rec_sq + gr.square().sum()
        valid = (rec_sq > eps) & (prior_sq > eps)
        coefficient = torch.where(valid, strength * dot.clamp(max=0) / rec_sq.clamp_min(eps), zero)
        if bool(valid):
            cosines.append(dot / (rec_sq.sqrt() * prior_sq.sqrt()))
            conflicts.append((dot < 0).float())
        original_sq = original_sq + prior_sq
        for i in active:
            if reference[i] is not None:
                prior[i] = prior[i] - coefficient.to(prior[i].dtype) * reference[i]
            projected_sq = projected_sq + prior[i].float().square().sum()
    for parameter, gr, gd in zip(parameters, rec, prior):
        parameter.grad = gd if gr is None else gr if gd is None else gr + gd
    return {
        'op_conflict_rate': torch.stack(conflicts).mean().detach() if conflicts else zero,
        'op_gradient_cosine': torch.stack(cosines).mean().detach() if cosines else zero,
        'op_prior_retention': torch.where(original_sq > eps,
            (projected_sq / original_sq.clamp_min(eps)).sqrt(), torch.ones_like(zero)).detach(),
    }


def _joint_soft_dual(gram, dots, strengths, eps=1e-6):
    """Exact active-set solve of a <=2-reference nonnegative soft dual.

    min_{lambda>=0} .5 lambda' (Gram + diag((1-eta)/eta)) lambda
                         + dots' lambda.
    This is symmetric in the references; there is no sequential projection.
    For a single reference it removes eta of the conflicting component.
    """
    count = len(dots)
    if count == 0:
        return dots
    if count > 2:
        raise ValueError('Light OP coordination supports at most two references')
    output_dtype = dots.dtype
    # Only a 1x1/2x2 system: double precision avoids cancellation when the
    # combined synthetic-prior gradient is much larger than a reference.
    gram, dots, strengths = gram.double(), dots.double(), strengths.double()
    eta = strengths.clamp(min=eps, max=1-eps)
    matrix = gram + torch.diag((1-eta) / eta) + eps * torch.eye(count, device=gram.device, dtype=gram.dtype)
    tolerance = eps * (1 + dots.abs().max())
    for bits in range(1 << count):
        active = [i for i in range(count) if bits & (1 << i)]
        solution = torch.zeros_like(dots)
        if active:
            idx = torch.tensor(active, device=gram.device)
            solution[idx] = torch.linalg.solve(matrix[idx][:, idx], -dots[idx])
        residual = matrix @ solution + dots
        inactive = [i for i in range(count) if i not in active]
        if bool((solution >= -tolerance).all()) and (not inactive or bool((residual[inactive] >= -tolerance).all())):
            return solution.clamp_min(0).to(output_dtype)
    raise RuntimeError('OP joint soft dual failed its KKT conditions')


def joint_observation_backward(base_loss, prior_loss, parameters, parameter_groups,
                               references, strength=0.5, adaptive=True, eps=1e-12):
    """Merge base + coordinated prior using scene/one-instance observations.

    references is [(loss, detached reliability), ...], at most two. Base keeps
    all direct observations and regularizers, irrespective of reliability.
    The caller chooses which sources belong to the prior. Only prior-connected
    parameter tensors receive the correction. This v3 path targets one GPU.
    """
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() != 1:
        raise ValueError('OP v3 is configured for single-GPU training only')
    if not 0 <= strength <= 1 or not 1 <= len(references) <= 2:
        raise ValueError('OP requires strength in [0,1] and one/two references')
    parameters = list(dict.fromkeys(p for p in parameters if p.requires_grad))
    if not parameters:
        raise ValueError('No trainable OP parameters')
    def grad(loss, retain=True):
        return (list(torch.autograd.grad(loss, parameters, retain_graph=retain, allow_unused=True))
                if loss.requires_grad else [None] * len(parameters))
    base = grad(base_loss)
    refs = [grad(loss) for loss, _ in references]
    prior = grad(prior_loss, retain=False)
    reliability = torch.stack([torch.as_tensor(c, device=base_loss.device, dtype=torch.float32).detach().clamp(0, 1)
                               for _, c in references])
    lookup = {id(p): i for i, p in enumerate(parameters)}
    groups, used = [], set()
    for members in parameter_groups.values():
        ids = list(dict.fromkeys(lookup[id(p)] for p in members if id(p) in lookup))
        if used.intersection(ids):
            raise ValueError('Overlapping OP groups')
        used.update(ids)
        groups.append(ids)
    groups.append([i for i in range(len(parameters)) if i not in used])
    zero = base_loss.detach().float().new_zeros(())
    old_sq, new_sq = zero.clone(), zero.clone()
    cosines, conflicts, etas, violations = [], [], [], []
    for group in groups:
        active = [i for i in group if prior[i] is not None]
        if not active:
            continue
        def dot(a, b):
            return sum(((a[i].float() * b[i].float()).sum() for i in active
                        if a[i] is not None and b[i] is not None), zero.clone())
        d_sq = dot(prior, prior)
        norm_sq = torch.stack([dot(g, g) for g in refs])
        keep = (norm_sq > eps) & (reliability > 0)
        old_sq = old_sq + d_sq
        if not bool(d_sq > eps) or not bool(keep.any()) or strength == 0:
            new_sq = new_sq + d_sq
            continue
        indices = keep.nonzero().flatten().tolist()
        rr = [refs[j] for j in indices]
        norms = norm_sq[indices].sqrt()
        dots = torch.stack([dot(g, prior) for g in rr]) / norms
        gram = torch.stack([torch.stack([dot(a, b) for b in rr]) for a in rr]) / (norms[:, None] * norms[None, :])
        cosine = (dots / d_sq.sqrt()).clamp(-1, 1)
        eta = strength * (reliability[indices] * (0.5 + 0.5 * (-cosine).clamp(0, 1))
                          if adaptive else torch.ones_like(cosine))
        if not bool(torch.isfinite(gram).all() & torch.isfinite(dots).all() & torch.isfinite(eta).all()):
            raise RuntimeError('Nonfinite OP reference/prior gradients')
        coefficients = _joint_soft_dual(gram, dots, eta) / norms
        for i in active:
            correction = sum((coefficients[j].to(prior[i]) * g[i] for j, g in enumerate(rr)
                              if g[i] is not None), torch.zeros_like(prior[i]))
            prior[i] = prior[i] + correction
        new_sq = new_sq + dot(prior, prior)
        post = torch.stack([dot(g, prior) for g in rr]) / norms
        cosines.extend(cosine.unbind());conflicts.extend((cosine < 0).float().unbind())
        etas.extend(eta.unbind());violations.extend((post < -1e-6).float().unbind())
    for p, gb, gp in zip(parameters, base, prior):
        p.grad = gp if gb is None else gb if gp is None else gb + gp
    mean = lambda items: torch.stack(items).mean().detach() if items else zero
    return {'op_conflict_rate': mean(conflicts), 'op_gradient_cosine': mean(cosines),
            'op_effective_strength': mean(etas), 'op_soft_violation_rate': mean(violations),
            'op_reference_count': zero.new_tensor(len(references)),
            'op_prior_retention': torch.where(old_sq > eps, (new_sq/old_sq.clamp_min(eps)).sqrt(), torch.ones_like(zero)).detach()}
