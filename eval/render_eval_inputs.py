"""Select rendering experiments and validate view IDs before loading metric models."""
from datetime import datetime
from pathlib import Path
import re


def rendering_runs(exp_root, dataset_type, run_dir=None):
    exp_root = Path(exp_root)
    marker = f'_{dataset_type}_'
    if run_dir:
        run = Path(run_dir).expanduser().resolve()
        if not run.is_dir() or marker not in run.parent.name:
            raise ValueError(
                f'--run_dir must be an experiment run inside a *{marker}SCAN directory: {run}'
            )
        return [(run.parent.name.split('_')[-1], run)]

    def chronological_key(run):
        match = re.search(r'(\d{4}(?:_\d{2}){5})$', run.name)
        if match:
            timestamp = datetime.strptime(match.group(1), '%Y_%m_%d_%H_%M_%S').timestamp()
        else:
            timestamp = run.stat().st_mtime
        return timestamp, run.name

    selected = []
    for experiment in sorted(exp_root.iterdir()):
        if not experiment.is_dir() or marker not in experiment.name:
            continue
        candidates = [run for run in experiment.iterdir() if run.is_dir()]
        if not candidates:
            raise ValueError(f'No experiment runs found: {experiment}')
        run = max(candidates, key=chronological_key)
        selected.append((experiment.name.split('_')[-1], run))
    if not selected:
        raise ValueError(f'No {dataset_type} experiments found under {exp_root}')
    return selected


def rendering_image_paths(plots_dir, target_dir):
    """Require both render types and the complete GT view set; never skip missing views."""
    plots = Path(plots_dir)
    target = Path(target_dir)
    target_paths = sorted(target.glob('*_rgb.png'))
    if not target_paths:
        raise ValueError(f'No GT RGB images found: {target}')

    def indexed(paths, prediction):
        views = {}
        for path in paths:
            token = path.stem.rsplit('_', 1)[-1] if prediction else path.stem.split('_')[0]
            try:
                view = int(token)
            except ValueError as exc:
                raise ValueError(f'Cannot parse view ID from image: {path}') from exc
            if view in views:
                raise ValueError(f'Duplicate view ID {view}: {views[view]}, {path}')
            views[view] = path
        return views

    expected = indexed(target_paths, False)
    groups = []
    for name in ('eval_nerf_rgb', 'eval_nvdiffrast_rgb'):
        directory = plots / name
        if not directory.is_dir():
            raise FileNotFoundError(
                f'Missing rendering directory: {directory}. '
                'Use --run_dir to select the completed texture experiment.'
            )
        predicted = indexed(sorted(directory.glob('*.png')), True)
        if not predicted or predicted.keys() != expected.keys():
            raise ValueError(
                f'View ID mismatch in {directory}: '
                f'missing={sorted(expected.keys() - predicted.keys())}, '
                f'extra={sorted(predicted.keys() - expected.keys())}'
            )
        groups.append([predicted[view] for view in sorted(expected)])
    return tuple(groups)
