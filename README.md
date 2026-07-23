# Local Multi-Module Facial Recognition & Person Identification Pipeline

A computer vision system combining facial recognition, liveness/anti-spoofing detection, super-resolution, and person re-identification (Re-ID) — designed to run **entirely locally on Google Colab, with no external API dependencies.**

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
