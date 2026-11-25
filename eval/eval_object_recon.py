"""Evaluate the complete GT instance set, including failed predictions."""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
import numpy as np
from evaluate_single_scene import failed_prediction_metrics


def read_txt_to_dict(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        key, value = line.split(': ', 1)
        result[key] = float(value)
    if set(result) != set(failed_prediction_metrics()):
        raise ValueError(f'Incomplete metrics: {path}')
    if any(np.isnan(value) for value in result.values()):
        raise ValueError(f'NaN metrics: {path}; rerun using the corrected evaluator')
    return result


def write_metrics(path, metrics):
    Path(path).write_text(''.join(f'{key}: {value}\n' for key, value in metrics.items()))


def mean_metrics(results):
    if not results:
        raise ValueError('No objects/scenes to average')
    keys = set(failed_prediction_metrics())
    if any(set(result) != keys for result in results):
        raise ValueError('Inconsistent metric keys')
    return {key: float(np.mean([result[key] for result in results])) for key in failed_prediction_metrics()}


def expected_objects(data_root, scan_id):
    gt_root = data_root / 'GTmesh' / f'scan{scan_id}'
    ids = sorted(int(match.group(1)) for p in gt_root.glob('obj_*.ply')
                 if (match := re.fullmatch(r'obj_(\d+)\.ply', p.name)) and int(match.group(1)) > 0)
    if not ids:
        raise ValueError(f'No ground-truth foreground objects in {gt_root}')
    mapping = data_root / f'scan{scan_id}' / 'instance_id.json'
    if mapping.exists():
        expected = list(range(1, len(json.loads(mapping.read_text())) + 1))
        if ids != expected:
            raise ValueError(f'GT objects {ids} disagree with training label mapping {expected}')
    return ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset_type', required=True)
    parser.add_argument('--root_path', default='./')
    args = parser.parse_args()
    root = Path(args.root_path).resolve()
    exp_root, data_root = root / 'exps', root / 'data' / args.dataset_type
    scene_results = []
    evaluator = Path(__file__).with_name('evaluate_single_scene.py')
    for experiment in sorted(exp_root.iterdir()):
        if not experiment.is_dir() or f'_{args.dataset_type}_' not in experiment.name:
            continue
        scan_id = experiment.name.split('_')[-1]
        runs = sorted(p for p in experiment.iterdir() if p.is_dir() and (p / 'plots').is_dir())
        if not runs:
            raise ValueError(f'No run with plots: {experiment}')
        plots = runs[-1] / 'plots'
        epochs = [int(m.group(1)) for p in plots.iterdir()
                  if (m := re.fullmatch(r'surface_(\d+)_(?:\d+|whole)\.ply', p.name))]
        if not epochs:
            raise ValueError(f'No exported reconstruction: {plots}')
        epoch = max(epochs)
        ids = expected_objects(data_root, scan_id)
        destination = plots / 'eval_obj'
        destination.mkdir(exist_ok=True)
        results, missing, zero_scores = [], 0, 0
        camera_file = 'cameras.npz' if args.dataset_type == 'replica' else 'eval_total_cameras.npz'
        for obj_id in ids:
            prediction = plots / f'surface_{epoch}_{obj_id}.ply'
            gt = data_root / 'GTmesh' / f'scan{scan_id}' / f'obj_{obj_id}.ply'
            output = destination / f'obj_{obj_id}_metrics.txt'
            if not prediction.exists():
                metrics = failed_prediction_metrics()
                write_metrics(output, metrics)
                missing += 1
                print(f'MISSING prediction: {prediction}; counted as failed')
            else:
                # A failed subprocess stops evaluation; stale output is never read.
                subprocess.run([sys.executable, str(evaluator), '--input_mesh', str(prediction),
                                '--gt_mesh', str(gt), '--input_scalemat', str(data_root / f'scan{scan_id}' / camera_file),
                                '--output_mesh', str(destination / f'fuse_obj_{obj_id}.ply'),
                                '--output_txt', str(output), '--none_fuse'], check=True, cwd=root)
                metrics = read_txt_to_dict(output)
            zero_scores += int(metrics['F-score'] == 0)
            results.append(metrics)
        averaged = mean_metrics(results)
        write_metrics(destination / 'metrics.txt', averaged)
        (destination / 'counts.txt').write_text(f'Expected objects: {len(ids)}\nMissing predictions: {missing}\nZero F-score objects (including missing): {zero_scores}\n')
        print(f'{experiment.name}: expected={len(ids)}, missing={missing}, zero-F={zero_scores}, metrics={averaged}')
        scene_results.append(averaged)
    total = mean_metrics(scene_results)
    write_metrics(exp_root / f'total_obj_metrics_{args.dataset_type}.txt', total)
    print(f'Dataset {args.dataset_type}, scenes={len(scene_results)} (macro scene mean): {total}')


if __name__ == '__main__':
    main()
