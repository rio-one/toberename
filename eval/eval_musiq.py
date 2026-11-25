import argparse
from pathlib import Path

from render_eval_inputs import rendering_runs, rendering_image_paths


NAME_TO_HANDLE = {
    'spaq': 'https://tfhub.dev/google/musiq/spaq/1',
    'koniq': 'https://tfhub.dev/google/musiq/koniq-10k/1',
    'paq2piq': 'https://tfhub.dev/google/musiq/paq2piq/1',
    'ava': 'https://tfhub.dev/google/musiq/ava/1',
}


def get_scene_musiq(image_paths, predict_fn, tf):
    total = 0.0
    for path in image_paths:
        image_bytes = path.read_bytes()
        prediction = predict_fn(tf.constant(image_bytes))
        total += prediction['output_0'].numpy().item()
    return total / len(image_paths)


def main():
    parser = argparse.ArgumentParser(description='Evaluate MUSIQ on matching rendering views.')
    parser.add_argument('--dataset_type', required=True)
    parser.add_argument('--root_path', default='./')
    parser.add_argument('--run_dir', help='Exact geometry/texture experiment directory, containing plots/.')
    parser.add_argument('--output_txt', help='Result file; defaults to plots/musiq.txt with --run_dir.')
    args = parser.parse_args()
    root = Path(args.root_path)
    exp_root = root / 'exps'

    # Resolve and validate all inputs before importing TensorFlow or loading MUSIQ.
    inputs = []
    for scan_id, run in rendering_runs(exp_root, args.dataset_type, args.run_dir):
        target = root / 'data' / args.dataset_type / f'scan{scan_id}' / 'eval-novel-views/rgb'
        nerf_paths, mesh_paths = rendering_image_paths(run / 'plots', target)
        print(f'[MUSIQ input] {run.resolve()}; NeRF={len(nerf_paths)}, nvdiffrast={len(mesh_paths)}',
              flush=True)
        inputs.append((run, nerf_paths, mesh_paths))

    import tensorflow as tf
    import tensorflow_hub as hub

    selected_model = 'koniq'
    model_handle = NAME_TO_HANDLE[selected_model]
    model = hub.load(model_handle)
    predict_fn = model.signatures['serving_default']
    print(f'loaded model {selected_model} ({model_handle})')
    results = {'nerf_rgb_musiq': 0.0, 'nvdiff_rgb_musiq': 0.0}
    for run, nerf_paths, mesh_paths in inputs:
        results['nerf_rgb_musiq'] += get_scene_musiq(nerf_paths, predict_fn, tf)
        results['nvdiff_rgb_musiq'] += get_scene_musiq(mesh_paths, predict_fn, tf)
    results = {key: value / len(inputs) for key, value in results.items()}
    summary = f'dataset: {args.dataset_type}, total scene: {len(inputs)}\n'
    summary += ''.join(f'{key}: {value}\n' for key, value in results.items())
    print(summary, end='')
    output = (Path(args.output_txt) if args.output_txt else
              inputs[0][0] / 'plots/musiq.txt' if args.run_dir else
              exp_root / f'musiq_{args.dataset_type}.txt')
    with output.open('w') as stream:
        stream.write(''.join(f'input run: {run.resolve()}\n' for run, _, _ in inputs))
        stream.write(summary)
    print(f'Saved MUSIQ results: {output.resolve()}')


if __name__ == '__main__':
    main()
