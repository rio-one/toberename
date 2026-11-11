"""Light smoke only: no diffusion downloads, training runs, mesh exports or hashes."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / '.oprecon_cache'
for name, folder in {
    'MPLCONFIGDIR': 'matplotlib', 'TORCH_HOME': 'torch',
    'TORCH_EXTENSIONS_DIR': 'torch_extensions', 'TRITON_CACHE_DIR': 'triton',
    'CUDA_CACHE_PATH': 'cuda', 'XDG_CACHE_HOME': 'xdg', 'HF_HOME': 'huggingface',
    'HF_HUB_CACHE': 'huggingface/hub', 'TMPDIR': 'tmp', 'TMP': 'tmp', 'TEMP': 'tmp',
}.items():
    path = CACHE / folder
    path.mkdir(parents=True, exist_ok=True)
    os.environ[name] = str(path)
sys.dont_write_bytecode = True
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  DIFFUSERS_OFFLINE='1', WANDB_MODE='disabled')
if Path(sys.prefix).name != 'oprecon':
    raise RuntimeError('Run this smoke test with the oprecon conda interpreter')
sys.path.insert(0, str(ROOT / 'code'))

import ast
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from pyhocon import ConfigFactory
from datasets.scene_dataset import SceneDatasetDN_segs
from model.loss import DPReconLoss
from utils.op_gradients import observation_protected_backward
from utils.op_sampling import InstanceBoundarySampler


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def gradient_contract(device):
    p = nn.Parameter(torch.tensor([1., 1.], device=device))
    unused = nn.Parameter(torch.ones(1, device=device))
    def run(gr, gd, strength=1.):
        stats = observation_protected_backward((p * gr).sum(), (p * gd).sum(),
            [p, unused], {'test': [p]}, strength=strength)
        check(unused.grad is None, 'Unused parameter received a gradient')
        return p.grad.clone(), stats
    gr = torch.tensor([1., 0.], device=device)
    grad, stats = run(gr, torch.tensor([-1., 2.], device=device))
    torch.testing.assert_close(grad, torch.tensor([1., 2.], device=device))
    check(stats['op_conflict_rate'].item() == 1., 'Conflict was not detected')
    check(stats['op_prior_retention'].item() > 0., 'Orthogonal completion was discarded')
    grad, _ = run(gr, gr)
    torch.testing.assert_close(grad, 2 * gr)
    grad, _ = run(torch.zeros_like(gr), gr)
    torch.testing.assert_close(grad, gr)
    grad, _ = run(gr, -gr, strength=0.)
    torch.testing.assert_close(grad, torch.zeros_like(gr))
    # A prior that does not reach a parameter must not acquire a new update there.
    q = nn.Parameter(torch.ones(1, device=device))
    observation_protected_backward(p.sum() + q.sum(), -p.sum(),
        [p, q], {'shared': [p, q]})
    torch.testing.assert_close(q.grad, torch.ones_like(q))
    print('PASS gradient projection: conflicting/aligned/zero/disjoint/disabled')


def runner_backward():
    # Exercise the actual trainer method without initializing diffusion, DDP,
    # experiment directories or the full training runner.
    tree = ast.parse((ROOT / 'code/training/DPRecon_train.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DPReconTrainRunner')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_op_backward')
    namespace = {'observation_protected_backward': observation_protected_backward}
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<actual OP trainer hook>', 'exec'), namespace)
    return namespace['_op_backward']


def stage_updates(device, data, conf):
    hook = runner_backward()
    class Texture(nn.Module):
        def __init__(self):
            super().__init__()
            self.pos_encoding = nn.Linear(3, 8)
            self.layers = nn.Sequential(nn.Tanh(), nn.Linear(8, 3), nn.Sigmoid())
        def forward(self, x):
            return self.layers(self.pos_encoding(x))
    field = nn.Sequential(nn.Linear(3, 12), nn.Tanh(), nn.Linear(12, 3)).to(device)
    color = nn.Sequential(nn.Linear(3, 8), nn.Tanh(), nn.Linear(8, 3), nn.Sigmoid()).to(device)
    texture = Texture().to(device)
    model = SimpleNamespace(implicit_network=field, mesh_color_network=texture)
    parameters = list(field.parameters()) + list(color.parameters()) + list(texture.parameters())
    optimizer = torch.optim.Adam(parameters, lr=1e-3)
    runner = SimpleNamespace(model=SimpleNamespace(module=model), Grid_MLP=False,
                             optimizer=optimizer, op_projection_strength=1.)
    loss_fn = DPReconLoss(**conf.get_config('loss'))
    x = torch.randn(256, 3, device=device, requires_grad=True)
    raw = field(x)
    sdf = raw[:, :1]
    normals = torch.autograd.grad(sdf.sum(), x, create_graph=True)[0]
    rgb = color(x + normals)
    output = dict(iter_step=35001, rgb_values=rgb,
        depth_values=torch.sigmoid(sdf) + 0.5, normal_map=normals,
        sdf=torch.cat([sdf - 2, sdf + 2], dim=-1),
        grad_theta=normals, grad_theta_nei=normals * 1.01,
        object_opacity=torch.sigmoid(raw), sample_sdf=raw,
        sample_minsdf=raw.min(dim=-1, keepdim=True).values,
        # A tiny normal-based SDS surrogate exercises the higher-order SDF
        # path. The pretrained diffusion model is intentionally not loaded.
        loss_rgb_sd=-(normals.square().sum() + 0.1 * sdf.square().sum()))
    gt = {k: v[:256].unsqueeze(0) for k, v in data.items()
          if k in ['rgb', 'depth', 'normal', 'mask', 'rgb_sampling_weight']}
    gt['segs'] = torch.zeros(1, 256, 1, dtype=torch.long)
    losses = loss_fn(output, gt, call_reg=True)
    rec_prior_grads = torch.autograd.grad(losses['reconstruction_loss'], normals,
                                        retain_graph=True, allow_unused=True)
    check(rec_prior_grads[0] is not None, 'Real geometry reconstruction graph is disconnected')
    before = next(field.parameters()).detach().clone()
    optimizer.zero_grad(set_to_none=True)
    stats = hook(runner, losses['reconstruction_loss'], losses['sds_loss'], 'geometry')
    check(all(torch.isfinite(p.grad).all() for p in parameters if p.grad is not None),
          'Non-finite geometry gradient')
    optimizer.step()
    check(not torch.equal(before, next(field.parameters()).detach()), 'Geometry did not update')
    print('PASS geometry: actual DPReconLoss + trainer hook + higher-order gradient + Adam step',
          {k: round(v.item(), 5) for k, v in stats.items()})
    for p in list(field.parameters()) + list(color.parameters()):
        p.requires_grad_(False)
    optimizer.zero_grad(set_to_none=True)
    fixed_geometry = next(field.parameters()).detach().clone()
    before = next(texture.parameters()).detach().clone()
    pred = texture(torch.randn(16, 3, device=device))
    rec = (pred - .2).square().mean()
    prior = 0.1 * (pred - .8).square().mean()
    stats = hook(runner, rec, prior, 'texture')
    check(all(torch.isfinite(p.grad).all() for p in texture.parameters()), 'Non-finite texture gradient')
    optimizer.step()
    check(not torch.equal(before, next(texture.parameters()).detach()), 'Texture did not update')
    torch.testing.assert_close(next(field.parameters()), fixed_geometry)
    print('PASS texture: actual trainer hook + Adam step; frozen geometry preserved',
          {k: round(v.item(), 5) for k, v in stats.items()})


def ddp_contract(device):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    import tempfile
    with tempfile.TemporaryDirectory(dir=CACHE / 'tmp') as work:
        dist.init_process_group('gloo', init_method='file://' + str(Path(work) / 'store'),
                                rank=0, world_size=1)
        try:
            model = DistributedDataParallel(nn.Linear(2, 2).to(device),
                device_ids=[device.index or 0], find_unused_parameters=True)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            with model.no_sync():
                pred = model(torch.ones(2, 2, device=device))
                observation_protected_backward(pred.square().mean(), -pred.mean(),
                    list(model.parameters()), {'ddp': list(model.parameters())})
            check(all(p.grad is not None and torch.isfinite(p.grad).all()
                      for p in model.parameters()), 'DDP no_sync hook gradients')
            optimizer.step()
            print('PASS single-rank DDP no_sync + protected backward + optimizer step')
        finally:
            dist.destroy_process_group()


def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    np.random.seed(42)
    print('Interpreter:', sys.executable)
    for rel in ['code/utils/op_gradients.py', 'code/utils/op_sampling.py',
                'code/datasets/scene_dataset.py', 'code/model/loss.py',
                'code/training/DPRecon_train.py']:
        ast.parse((ROOT / rel).read_text(), filename=rel)
    for path in sorted((ROOT / 'code/confs').glob('*_grid.conf')):
        cfg = ConfigFactory.parse_file(str(path))
        check(cfg.get_bool('train.op_geometry_conflict') and cfg.get_bool('train.op_texture_conflict'), 'Stage switch default')
        check(cfg.get_bool('dataset.op_balanced_sampling'), 'Sampling switch default')
        check(cfg.get_string('train.expname').startswith('oprecon_'), 'Experiment names are not isolated')
    print('PASS Python syntax and all three default configurations')
    conf = ConfigFactory.parse_file(str(ROOT / 'code/confs/replica_grid.conf'))
    ds_conf = dict(conf.get_config('dataset'))
    ds_conf['data_root_dir'] = str(ROOT / 'data')
    dataset = SceneDatasetDN_segs(scan_id=1, **ds_conf)
    dataset.change_sampling_idx(256)
    for view in [0, min(1, len(dataset) - 1)]:
        _, sample, gt = dataset[view]
        check(sample['uv'].shape == (256, 2), 'Wrong sampled ray count')
        weights = gt['rgb_sampling_weight']
        check(weights.shape == (256, 1) and torch.isfinite(weights).all() and (weights > 0).all(), 'RGB weights')
        check(gt['segs'].min() >= 0 and gt['segs'].max() < len(dataset.label_mapping), 'Label range')
    batch = dataset.collate_fn([dataset[0]])
    check(batch[2]['rgb_sampling_weight'].shape == (1, 256, 1), 'Collate weights')
    background = InstanceBoundarySampler([torch.zeros(16, 1)], (4, 4))
    idx, weights = background.sample(0, 8)
    torch.testing.assert_close(weights, torch.ones_like(weights))
    check(len(idx) == 8, 'Empty-pool fallback')
    sampler = dataset.op_sampler
    dataset.op_sampler = None
    dataset.change_sampling_idx(256)
    _, _, plain = dataset[0]
    check('rgb_sampling_weight' not in plain, 'Sampling-off did not restore original path')
    dataset.op_sampler = sampler
    dataset.change_sampling_idx(-1)
    _, _, full = dataset[0]
    check(full['rgb'].shape[0] == dataset.total_pixels and 'rgb_sampling_weight' not in full,
          'Full-image texture path changed')
    print('PASS real scan1 samples, weights, collate, fallback, sampling-off and full-image paths')
    gradient_contract(torch.device('cpu'))
    if not torch.cuda.is_available():
        print('SKIP CUDA stage updates: no visible GPU; CPU checks passed')
        return
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    gradient_contract(device)
    ddp_contract(device)
    stage_updates(device, {name: value[0] for name, value in batch[2].items()}, conf)
    print('Peak CUDA allocated MiB:', round(torch.cuda.max_memory_allocated() / 1024**2, 2))
    print('SMOKE PASSED. No pretrained diffusion load, full training, meshes or hash verification.')


if __name__ == '__main__':
    main()
