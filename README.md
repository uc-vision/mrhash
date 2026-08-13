# mrhash directional TSDF

This CropVision fork provides sparse, GPU-resident directional TSDF fusion and native surface extraction. Each sparse block is keyed by both its spatial coordinate and one of the six signed Cartesian directions. Voxels retain additive signed-distance and weight statistics so the field can later support solid-volume classification.

The active API intentionally does not expose regular TSDF, TUDF, or Gaussian optimization. Surface output includes unique surface voxels and an independent triangle mesh built directly from directional zero crossings. The mesh path filters and intersects per-direction marching-cubes cases into at most two opposite components per cell, computes voted weighted offsets on their shared grid edges, and triangulates the resulting cell cases. It never triangulates the extracted surface voxels.

## Build

The parent workspace owns the Pixi environment and CUDA wheel workflow. Read `../../build-cuda/README.md` before building or publishing a wheel. Do not install the package with pip or conda directly.

For a local CUDA 12.9 / Python 3.12 trial wheel on a Blackwell GPU:

```sh
cd ../../build-cuda
CUDA_ARCHITECTURES=120 pixi run python -m build \
  --wheel --no-isolation --outdir /tmp/mrhash-directional-wheel ../build/mrhash
```

## Python API

```python
import numpy as np
import torch

from mrhash.src.pygeowrapper import DirectionalVoxelMap

voxel_map = DirectionalVoxelMap(
  voxel_size=0.001,
  minimum_depth=0.1,
  maximum_depth=1.5,
)
voxel_map.set_camera(focal_x, focal_y, center_x, center_y, rows, columns)

for depth, translation, quaternion_xyzw in frames:
  voxel_map.set_pose(
    np.asarray(translation, dtype=np.float32),
    np.asarray(quaternion_xyzw, dtype=np.float32),
  )
  voxel_map.integrate_depth_cuda(depth.contiguous().to(device='cuda', dtype=torch.float32))

surface = voxel_map.extract_surface()
indices = np.asarray(surface.voxels.indices)
confidence = np.asarray(surface.voxels.confidence)
normals = np.asarray(surface.voxels.normals)
vertices = np.asarray(surface.mesh.vertices)
faces = np.asarray(surface.mesh.faces)
```

`extract_surface()` returns globally unique integer surface-voxel indices with directional vote confidence and normals under `surface.voxels`. `surface.mesh` contains the direct field mesh as globally welded floating-point vertices and indexed triangle faces. Neither output classifies or fills a solid volume. `serialize_grid()` and `deserialize_grid()` preserve the signed directional fields for later processing.

The workspace entry point is `voxelize-tsdf`; it renders depth from the Gaussian model, fuses it through this API, colors the extracted surface voxels, validates them, and opens the voxel viewer.

## Tests

Configure `MRHASH_BUILD_TESTS=ON`, build `test_directional_tsdf`, and run it from the build-cuda Pixi environment. The focused suite covers directional hash identity, signed plane crossings, watertight direct meshing of a closed field, and exact additive statistics across GPU/host streaming.

## Upstream

This implementation retains the sparse hash and streaming foundation from MrHash:

```text
@article{10.1145/3777909,
author = {De Rebotti, Lorenzo and Giacomini, Emanuele and Grisetti, Giorgio and Di Giammarino, Luca},
title = {Resolution Where It Counts: Hash-based GPU-Accelerated 3D Reconstruction via Variance-Adaptive Voxel Grids},
year = {2025},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
issn = {0730-0301},
url = {https://doi.org/10.1145/3777909},
doi = {10.1145/3777909},
journal = {ACM Trans. Graph.},
keywords = {Surface Reconstruction, Novel View Synthesis, Gaussian Splatting}}
```
