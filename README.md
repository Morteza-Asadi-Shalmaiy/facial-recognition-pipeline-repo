# facial-recognition-pipeline 🕵️
![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-005CED?logo=onnx&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Colab](https://img.shields.io/badge/Run%20in-Colab-F9AB00?logo=googlecolab&logoColor=white)

> **Results at a glance:** 4-stage fallback pipeline — spoof check → face recognition →
> super-resolution retry → person Re-ID · Zero external APIs, fully local Colab GPU
> execution · Designed around realistic CCTV conditions (low-res, angled, distant faces)

A **local, multi-module identity-verification pipeline** combining facial recognition,
liveness/anti-spoofing detection, super-resolution, and person re-identification — each
stage only activating if the one before it can't confidently resolve an identity. Built
entirely on local/Colab GPU resources with no cloud recognition APIs (no AWS Rekognition,
no Azure Face), and designed from the ground up around realistic surveillance conditions:
low resolution, distant or angled faces, and presentation attacks.

Built as a portfolio project to demonstrate an end-to-end identity-verification pipeline suitable for realistic CCTV-style scenarios.

## Results for [First Module](https://github.com/Morteza-Asadi-Shalmaiy/facial-recognition-pipeline-repo/tree/main/modules/face-recognition) the base of our FR

### Before / After

| Before | After |
|--------|-------|
| ![Before](https://raw.githubusercontent.com/Morteza-Asadi-Shalmaiy/facial-recognition-pipeline/main/assets/test/test-image-04.jpg) | ![After](https://raw.githubusercontent.com/Morteza-Asadi-Shalmaiy/facial-recognition-pipeline/main/assets/results/test-image-04.jpg) |

### Demo Video

| Before | After |
|--------|-------|
| ![Demo](https://raw.githubusercontent.com/Morteza-Asadi-Shalmaiy/facial-recognition-pipeline/refs/heads/main/assets/video-test-01%20(1).gif) | ![Demo](https://raw.githubusercontent.com/Morteza-Asadi-Shalmaiy/facial-recognition-pipeline/refs/heads/main/assets/video-test-01.gif) |

## Pipeline Architecture

```
Input (image/video)
      |
      v
[1] Spoof Check (Liveness / Anti-Spoofing)
      |  (reject if spoof detected)
      v
[2] Face Recognition (InsightFace buffalo_l)
      |  (if match confidence too low)
      v
[3] Super-Resolution Fallback (CodeFormer / GFPGAN / Real-ESRGAN)
      |  (re-attempt recognition on restored face)
      v
[4] Person Re-ID Fallback (OSNet + YOLOv8)
      |  (if face still unresolved, match by full-body appearance)
      v
Final identity decision + case file / tracking output
```

## Modules

| Module | Status | Description |
|---|---|---|
| [`face-recognition`](./modules/face-recognition) | ✅ Complete | Identity matching via InsightFace `buffalo_l` + cosine similarity |
| [`liveness-anti-spoofing`](./modules/liveness-anti-spoofing) | 🚧 In progress | - |
| [`super-resolution`](./modules/super-resolution) | 🚧 In progress | - |
| [`person-reid`](./modules/person-reid) | 🚧 In progress | - |

Each module folder contains its own README with technical details, challenges solved, and results.

## Key Design Constraints

- **Fully local execution** — no cloud recognition APIs (e.g., AWS Rekognition, Azure Face); all models run on local/Colab GPU resources
- **Fallback-based architecture** — each stage only activates if the previous stage can't confidently resolve an identity, balancing speed and robustness
- **Realistic CCTV conditions in mind** — low resolution, distant/angled faces, and potential presentation attacks all factored into module selection

## Tech Stack

`InsightFace` · `ONNX Runtime` · `OpenCV` · `MiniFASNet` · `MediaPipe` · `CodeFormer` · `GFPGAN` · `Real-ESRGAN` · `TorchReID (OSNet)` · `YOLOv8` · `NumPy` · `Google Colab (GPU runtime)`

## Repository Structure

```
.
├── README.md                          # This file
├── modules/
│   ├── face-recognition/
│   ├── liveness-anti-spoofing/
│   ├── super-resolution/
│   └── person-reid/
├── docs/                              # Architecture diagrams, presentation notes
└── assets/                            # Demo images/GIFs/screenshots
```

## Status

Actively in development. Modules are being finalized and documented individually before being wired into the full end-to-end pipeline.
