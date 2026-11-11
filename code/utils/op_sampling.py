"""Per-view instance/boundary sampling using training masks only."""
import cv2
import numpy as np
import torch


class InstanceBoundarySampler:
    def __init__(self, labels, img_res, instance_fraction=0.25,
                 boundary_fraction=0.25, boundary_width=5):
        if (instance_fraction < 0 or boundary_fraction < 0 or
                instance_fraction + boundary_fraction >= 1):
            raise ValueError('OP sampling needs a positive uniform fraction')
        if boundary_width < 1 or boundary_width % 2 == 0:
            raise ValueError('OP boundary_width must be a positive odd integer')
        self.labels = labels
        self.img_res = tuple(img_res)
        self.total_pixels = int(np.prod(img_res))
        self.instance_fraction = instance_fraction
        self.boundary_fraction = boundary_fraction
        self.boundary_width = boundary_width
        self._cache = {}

    def _pools(self, view):
        if view not in self._cache:
            labels = self.labels[view].reshape(self.img_res).numpy().astype(np.int64)
            flat = labels.reshape(-1)
            pools = [np.flatnonzero(flat == obj) for obj in np.unique(flat) if obj > 0]
            instance_prob = np.zeros(self.total_pixels, dtype=np.float32)
            for pool in pools:
                instance_prob[pool] = 1.0 / (len(pools) * len(pool))
            edge = np.zeros(self.img_res, dtype=np.uint8)
            different = labels[:, 1:] != labels[:, :-1]
            edge[:, 1:] |= different
            edge[:, :-1] |= different
            different = labels[1:, :] != labels[:-1, :]
            edge[1:, :] |= different
            edge[:-1, :] |= different
            edge = cv2.dilate(edge, np.ones((self.boundary_width, self.boundary_width), np.uint8))
            boundary = np.flatnonzero(edge.reshape(-1))
            self._cache[view] = pools, instance_prob, boundary, edge.reshape(-1).astype(bool)
        return self._cache[view]

    def sample(self, view, size):
        if size < 1:
            raise ValueError('OP sample size must be positive')
        pools, instance_prob, boundary, is_boundary = self._pools(view)
        n_instance = int(size * self.instance_fraction) if pools else 0
        n_boundary = int(size * self.boundary_fraction) if len(boundary) else 0
        n_uniform = size - n_instance - n_boundary
        parts = [np.random.randint(self.total_pixels, size=n_uniform)]
        if n_instance:
            objects = np.random.randint(len(pools), size=n_instance)
            draws = np.empty(n_instance, dtype=np.int64)
            for obj in np.unique(objects):
                selected = objects == obj
                draws[selected] = np.random.choice(pools[obj], int(selected.sum()), replace=True)
            parts.append(draws)
        if n_boundary:
            parts.append(np.random.choice(boundary, n_boundary, replace=True))
        indices = np.concatenate(parts)
        np.random.shuffle(indices)
        # Balance-heuristic importance weights preserve the full-image RGB
        # objective under fixed-count mixture sampling (sampling with replacement).
        probability = (np.full(size, n_uniform / size / self.total_pixels, dtype=np.float32)
                       + n_instance / size * instance_prob[indices])
        if n_boundary:
            probability += n_boundary / size * is_boundary[indices] / len(boundary)
        weights = (1.0 / self.total_pixels / probability).astype(np.float32)
        return torch.from_numpy(indices.copy()).long(), torch.from_numpy(weights[:, None])
