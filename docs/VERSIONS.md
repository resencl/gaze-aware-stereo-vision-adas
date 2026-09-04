# Version record

Last verified: 2026-09-04.

## Software environment

The project targets Python 3.11. Exact direct dependencies are pinned in the
repository requirements files:

| Component | Version | Scope |
|---|---:|---|
| Python | 3.11 | Target interpreter |
| DepthAI | 2.32.0.0 | OAK device pipelines and OpenVINO blobs |
| NumPy | 2.4.4 | Numerical operations |
| OpenCV Python | 4.13.0.92 | Geometry, visualization, and calibration |
| PyTorch | 2.11.0 | Gaze-model inference and training |
| TorchVision | 0.26.0 | ResNet architecture and preprocessing |
| Ultralytics | 8.4.41 | YOLOv8 instance-segmentation runtime |
| h5py | 3.14.0 | Training-data reader; training only |
| Pillow | 12.2.0 | Training image processing |

`requirements.txt` defines the runtime environment and
`training/requirements.txt` adds the training-only dependencies. Transitive
dependencies are not locked; use a resolved environment export if byte-for-byte
environment reconstruction is required.

## Model artifacts

Model artifact versions are independent of the package versions used to load
them. For example, the inspected `yolov8n-seg.pt` checkpoint records
Ultralytics `8.0.0.dev0`, while the runtime is pinned to Ultralytics `8.4.41`.

See [`models/manifest.json`](../models/manifest.json) for artifact versions,
interfaces, file sizes, and SHA-256 fingerprints.

## Hardware and streams

| Component | Configuration |
|---|---|
| Road camera | OAK-D LR |
| Driver camera | OAK-D Pro |
| Runtime output size | 640 x 360 pixels |
| OAK-D LR RGB rate | 30 FPS |
| OAK-D LR left/right mono rate | 30 FPS |
| OAK-D Pro RGB rate | 30 FPS |
| Stereo profile | DepthAI `HIGH_DENSITY` |

Device MXIDs are runtime arguments and are intentionally not versioned. The
device firmware version is not independently pinned; it is managed by the
pinned DepthAI release. Record firmware, operating system, GPU, and driver
versions alongside any new experimental run when those factors may affect
performance.

## Repository history

The repository uses Git commits as the implementation version. Report the full
commit hash with experimental results. The training code was introduced in
commit `860c84a`; the calibration generator was introduced in commit
`1f25ac3`.
