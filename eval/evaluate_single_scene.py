# adapted from https://github.com/zju3dv/manhattan_sdf
import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree
import trimesh
import torch
import glob
import os
import os
from tqdm import tqdm
from pathlib import Path
import argparse
import subprocess
import sys

os.environ['PYOPENGL_PLATFORM'] = 'egl'

def nn_correspondance(verts1, verts2):
    indices = []
    distances = []
    if len(verts1) == 0 or len(verts2) == 0:
        return indices, distances

    kdtree = KDTree(verts1)
    distances, indices = kdtree.query(verts2)
    distances = distances.reshape(-1)

    return distances, indices


def failed_prediction_metrics():
    # Infinite distance is an explicit failure, never an excluded/favorable sample.
    return {'Acc': float('inf'), 'Comp': float('inf'), 'Chamfer-L1': float('inf'),
            'Prec': 0.0, 'Recal': 0.0, 'F-score': 0.0,
            'Normal-Acc': 0.0, 'Normal-Comp': 0.0, 'Normal-Consistency': 0.0}


def load_mesh(path):
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = (trimesh.util.concatenate(tuple(mesh.geometry.values())) if mesh.geometry
                else trimesh.Trimesh(vertices=np.empty((0, 3)), faces=np.empty((0, 3), dtype=np.int64)))
    return mesh


def evaluate(mesh_pred, mesh_trgt, threshold=.05, down_sample=.02, normal_samples=200000):
    if len(mesh_trgt.vertices) == 0 or len(mesh_trgt.faces) == 0:
        raise ValueError('Ground-truth mesh is empty')
    if not np.isfinite(mesh_trgt.vertices).all():
        raise ValueError('Ground-truth vertices are nonfinite')
    if len(mesh_pred.vertices) == 0 or len(mesh_pred.faces) == 0:
        return failed_prediction_metrics()
    if not np.isfinite(mesh_pred.vertices).all():
        raise ValueError('Prediction vertices are nonfinite')
    if mesh_trgt.area <= 0:
        raise ValueError('Ground-truth mesh has zero area')
    if mesh_pred.area <= 0:
        return failed_prediction_metrics()
    pcd_trgt = o3d.geometry.PointCloud()
    pcd_pred = o3d.geometry.PointCloud()

    pcd_trgt.points = o3d.utility.Vector3dVector(mesh_trgt.vertices[:, :3])
    pcd_pred.points = o3d.utility.Vector3dVector(mesh_pred.vertices[:, :3])

    if down_sample:
        pcd_pred = pcd_pred.voxel_down_sample(down_sample)
        pcd_trgt = pcd_trgt.voxel_down_sample(down_sample)

    verts_pred = np.asarray(pcd_pred.points)
    verts_trgt = np.asarray(pcd_trgt.points)

    dist1, _ = nn_correspondance(verts_pred, verts_trgt)
    dist2, _ = nn_correspondance(verts_trgt, verts_pred)

    precision = np.mean((dist2 < threshold).astype('float'))
    recal = np.mean((dist1 < threshold).astype('float'))
    fscore = 2 * precision * recal / (precision + recal) if precision + recal > 0 else 0.0

    # normal consistency
    N = normal_samples
    if N < 1:
        raise ValueError("normal_samples must be positive")
    pointcloud_pred, idx = mesh_pred.sample(N, return_index=True)
    pointcloud_pred = pointcloud_pred.astype(np.float32)
    normal_pred = mesh_pred.face_normals[idx]

    pointcloud_trgt, idx = mesh_trgt.sample(N, return_index=True)
    pointcloud_trgt = pointcloud_trgt.astype(np.float32)
    normal_trgt = mesh_trgt.face_normals[idx]

    _, index1 = nn_correspondance(pointcloud_pred, pointcloud_trgt)
    _, index2 = nn_correspondance(pointcloud_trgt, pointcloud_pred)

    normal_acc = np.abs((normal_pred * normal_trgt[index2.reshape(-1)]).sum(axis=-1)).mean()
    normal_comp = np.abs((normal_trgt * normal_pred[index1.reshape(-1)]).sum(axis=-1)).mean()
    normal_consistency = (normal_acc + normal_comp) * 0.5

    metrics = {
        'Acc': np.mean(dist2),
        'Comp': np.mean(dist1),
        'Chamfer-L1': (np.mean(dist2) + np.mean(dist1)) / 2,
        'Prec': precision,
        'Recal': recal,
        'F-score': fscore,
        'Normal-Acc': normal_acc,
        'Normal-Comp': normal_comp,
        'Normal-Consistency': normal_consistency,
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Arguments to evaluate the scene.'
    )

    parser.add_argument('--input_mesh', type=str,  help='path to the mesh to be culled')
    parser.add_argument('--gt_mesh', type=str,  help='path to the gt mesh')
    parser.add_argument('--input_scalemat', type=str,  help='path to the scale mat')
    parser.add_argument('--traj', type=str, default=None,  help='path to the trajectory')
    parser.add_argument('--output_mesh', type=str,  help='path to the output mesh')
    parser.add_argument('--output_txt', type=str,  help='path to the output metrics txt')
    parser.add_argument("--none_fuse", default=False, action="store_true")
    args = parser.parse_args()

    output_mesh = args.output_mesh
    if args.none_fuse:
        temp_mesh = load_mesh(args.input_mesh)
        # transform to original coordinate system with scale mat
        if args.input_scalemat is not None:
            scalemat = np.load(args.input_scalemat)['scale_mat_0']
            temp_mesh.vertices = temp_mesh.vertices @ scalemat[:3, :3].T + scalemat[:3, 3]
        else:
            # print('not input scalemat')
            scalemat = np.eye(4)
            temp_mesh.vertices = temp_mesh.vertices @ scalemat[:3, :3].T + scalemat[:3, 3]
        temp_mesh.export(output_mesh)
    else:
        fusion_script = 'code/utils/fuse_mesh_replica.py' if args.traj is not None else 'code/utils/fuse_mesh_scannet_scannetpp.py'
        command = [sys.executable, fusion_script, '--input_mesh', args.input_mesh,
                   '--input_scalemat', args.input_scalemat, '--output_mesh', output_mesh]
        if args.traj is not None:
            command.extend(['--traj', args.traj])
        subprocess.run(command, check=True)
        print(f'export fuse mesh: {output_mesh}')

    fuse_mesh = load_mesh(output_mesh)
    gt_mesh = load_mesh(args.gt_mesh)

    metrics = evaluate(fuse_mesh, gt_mesh)
    print(metrics)

    # save metric to txt file
    with open(args.output_txt, 'w') as f:
        for k, v in metrics.items():
            out = f"{k}: {v}\n"
            f.write(out)


if __name__ == "__main__":
    main()
