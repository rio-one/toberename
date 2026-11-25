import os
import cv2
import torch
import argparse
import torch.nn as nn
from glob import glob
from piq import ssim
from piq import psnr
import lpips
from render_eval_inputs import rendering_runs, rendering_image_paths


class ReconMetrics(nn.Module):
    def __init__(self, lpips_net='vgg', batchsize=8):
        super().__init__()
        if batchsize < 1:
            raise ValueError('batchsize must be positive')
        self.batchsize = batchsize
        self.metrics = {
            "ssim": ssim,
            "psnr": psnr,
            "lpips": lpips.LPIPS(net=lpips_net),
        }

    def forward(self, preds, targets):
        r"""
        Input:
            preds: [B, C, H, W]
            targets: [B, C, H, W]
        Return:
            metrics: dict of metrics
        """
        if len(preds) == 0 or preds.shape != targets.shape:
            raise ValueError('Rendering metrics require nonempty matching tensors')
        metrics = {}
        with torch.no_grad():
            for name in self.metrics:
                total = 0.0
                for start in range(0, len(preds), self.batchsize):
                    pred = preds[start:start+self.batchsize]
                    target = targets[start:start+self.batchsize]
                    values = self.compute(pred, target, metric=name)
                    total += values.mean().item() * len(pred)
                metrics[name] = total / len(preds)
        return metrics

    def compute(self, preds, targets, metric='psnr'):
        if metric == 'lpips':
            # Images are [0,1]. LPIPS internally maps to [-1,1] with normalize.
            return self.metrics[metric](preds, targets, normalize=True)
        return self.metrics[metric](preds, targets, reduction='none')

    def metrics_name(self):
        return list(self.metrics.keys())

    def set_divice(self, device):
        self.metrics["lpips"].to(device)


def matched_image_paths(pred_dir, target_dir):
    def indexed(paths, prediction):
        result = {}
        for path in paths:
            name = os.path.basename(path)
            view = int(name.rsplit('_', 1)[-1].split('.')[0] if prediction else name.split('_')[0])
            if view in result:
                raise ValueError(f'Duplicate view ID {view}: {path}')
            result[view] = path
        return result
    predicted = indexed(glob(os.path.join(pred_dir, '*.png')), True)
    targets = indexed(glob(os.path.join(target_dir, '*_rgb.png')), False)
    if not predicted or predicted.keys() != targets.keys():
        raise ValueError(f'View ID mismatch: missing={sorted(targets.keys()-predicted.keys())}, extra={sorted(predicted.keys()-targets.keys())}')
    return [(predicted[view], targets[view]) for view in sorted(targets)]


def load_images(pred_dir, target_dir, device=None):
    pairs = matched_image_paths(pred_dir, target_dir)
    def load(path):
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f'Cannot read image: {path}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    preds = torch.stack([load(p) for p, _ in pairs])
    targets = torch.stack([load(t) for _, t in pairs])
    if preds.shape != targets.shape:
        raise ValueError('Predicted/GT image dimensions differ')
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    return preds.to(device), targets.to(device)


def eval_rendering(pred_dir, target_dir, lpips_net='vgg', batchsize=16):
    recon_metrics = ReconMetrics(lpips_net, batchsize)
    preds, targets = load_images(pred_dir, target_dir)
    recon_metrics.set_divice(preds.device)
    metrics = recon_metrics(preds, targets)
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Arguments to evaluate total scene reconstruction metrics.'
    )
    parser.add_argument('--dataset_type', type=str)
    parser.add_argument('--run_dir', help='Exact experiment run directory, containing plots/.')
    parser.add_argument('--pred_dir')
    parser.add_argument('--target_dir')
    parser.add_argument('--output_txt')
    parser.add_argument('--root_path', type=str, default='./')
    args = parser.parse_args()
    if args.run_dir and (args.pred_dir or args.target_dir):
        parser.error('--run_dir cannot be combined with --pred_dir/--target_dir')
    if args.pred_dir or args.target_dir:
        if not args.pred_dir or not args.target_dir:
            parser.error('Both --pred_dir and --target_dir are required')
        result = eval_rendering(args.pred_dir, args.target_dir)
        print(result)
        if args.output_txt:
            with open(args.output_txt, 'w') as stream:
                stream.write(''.join(f'{key}: {value}\n' for key, value in result.items()))
        return
    if not args.dataset_type:
        parser.error('--dataset_type is required in dataset mode')

    dataset_type = args.dataset_type
    root_path = args.root_path
    exp_root_path = os.path.join(root_path, 'exps')
    data_root_path = os.path.join(root_path, f'data/{dataset_type}')

    nerf_results = {}                       # for all scenes
    nvdiffrast_results = {}
    total_scene = 0

    inputs = []
    for scan_id, run in rendering_runs(exp_root_path, dataset_type, args.run_dir):
        exp_path = str(run / 'plots')
        target_dir = os.path.join(data_root_path, f'scan{scan_id}', 'eval-novel-views', 'rgb')
        nerf_paths, mesh_paths = rendering_image_paths(exp_path, target_dir)
        print(f'[Rendering input] {run.resolve()}; NeRF={len(nerf_paths)}, nvdiffrast={len(mesh_paths)}',
              flush=True)
        inputs.append((exp_path, target_dir))

    for exp_path, target_dir in inputs:

        # 1.eval nerf rgb
        test_name = 'eval_nerf_rgb'
        pred_dir = os.path.join(exp_path, test_name)
        nerf_metrics = eval_rendering(pred_dir, target_dir)

        for k, v in nerf_metrics.items():
            if k not in nerf_results:
                nerf_results[k] = v
            else:
                nerf_results[k] += v

        # 2.eval nvdiffrast rgb
        test_name = 'eval_nvdiffrast_rgb'
        pred_dir = os.path.join(exp_path, test_name)
        nvdiffrast_metrics = eval_rendering(pred_dir, target_dir)

        for k, v in nvdiffrast_metrics.items():
            if k not in nvdiffrast_results:
                nvdiffrast_results[k] = v
            else:
                nvdiffrast_results[k] += v

        total_scene += 1

    if total_scene == 0:
        raise ValueError('No scenes found')
    print(f'dataset: {dataset_type}, total scene: {total_scene}')

    print('nerf rendering results:')
    for k, v in nerf_results.items():
        print(f'nerf {k}: {v/total_scene}')
    total_nerf_results_txt_path = os.path.join(exp_root_path, f'nerf_rendering_{dataset_type}.txt')
    with open(total_nerf_results_txt_path, 'w') as f:
        for k, v in nerf_results.items():
            f.write(f'{k}: {v/total_scene}\n')

    print('nvdiffrast rendering results:')
    for k, v in nvdiffrast_results.items():
        print(f'nvdiffrast {k}: {v/total_scene}')
    total_nvdiffrast_results_txt_path = os.path.join(exp_root_path, f'nvdiffrast_rendering_{dataset_type}.txt')
    with open(total_nvdiffrast_results_txt_path, 'w') as f:
        for k, v in nvdiffrast_results.items():
            f.write(f'{k}: {v/total_scene}\n')




if __name__ == "__main__":
    main()
