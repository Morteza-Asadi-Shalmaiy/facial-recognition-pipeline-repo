# Face Recognition Module — Multi-Module CV Pipeline

## Overview

This module is the identity-recognition core of a larger computer vision pipeline that also includes liveness/anti-spoofing detection, super-resolution preprocessing, and person re-identification (ReID). The full system is designed to run **entirely locally on Google Colab, with no external API dependencies** — a deliberate constraint that shaped every implementation decision below.

The face recognition module handles:
- Building a searchable identity database from a set of reference images
- Matching detected faces against that database using embedding similarity
- Running inference on both static images and video, with annotated visual output

## Architecture & Approach

**Model:** InsightFace `buffalo_l` (ONNX Runtime backend, GPU-accelerated via CUDAExecutionProvider with CPU fallback)

**Pipeline flow:**
1. **Enrollment** — Iterate over a directory of reference face images, run detection, and store the L2-normalized embedding of the largest detected face per image (handles multi-face reference photos and skips unreadable files or images with no detectable face)
2. **Matching** — For each detected face at inference time, compute cosine similarity against every enrolled embedding and return the best match with a similarity score
3. **Inference modes:**
   - **Image mode:** single-frame detection + annotated output image
   - **Video mode:** frame-by-frame processing with `cv2.VideoWriter` output, progress logging every 100 frames, and per-frame structured results (frame index, matched name, similarity %, timestamp)

**Supporting utilities:**
- A localized (Iran/Tehran timezone, Jalali calendar) timestamping utility, reflecting a deployment target with regional requirements
- Google Drive integration for asset management within the Colab environment

## Technical Challenges Solved

- **Reference image robustness:** enrollment gracefully handles corrupt/unreadable files and images without a detectable face, rather than crashing the pipeline
- **Multi-face reference handling:** automatically selects the most prominent (largest bounding-box) face when a reference image contains more than one person
- **Local-only constraint:** no reliance on cloud recognition APIs (e.g., AWS Rekognition, Azure Face) — the entire matching pipeline runs on local/Colab GPU resources using open-weight models
- **Video throughput:** frame-by-frame identity annotation with live progress reporting, structured for downstream integration with the ReID and liveness modules in the full pipeline

## Tech Stack

`InsightFace` · `ONNX Runtime` · `OpenCV` · `NumPy` · `Google Colab (GPU runtime)`

## Suggested Next Steps for This Write-Up

To make this presentation-ready for interviews, consider adding once available:
- **Quantified results:** recognition accuracy, false-match rate, or similarity threshold tuning results on a test set
- **A confidence threshold decision:** the code currently accepts a `threshold` parameter but doesn't yet gate matches on it — worth deciding and documenting whether unmatched faces below a certain similarity should be forced to "Unknown"
- **Integration diagram:** once you share the other 3 modules, a single pipeline diagram (spoof check → face recognition → super-resolution fallback → ReID fallback) will tie this into the "why does this matter as a system" story interviewers respond well to

---
*This is module 1 of 4 in the full pipeline. Update this document as the liveness, super-resolution, and Re-ID modules are finalized.*

---

## Results

### Before / After

| Before | After |
|--------|-------|
| ![Before](https://raw.githubusercontent.com/Morteza-Asadi-Shalmaiy/facial-recognition-pipeline-repo/main/assets/test/test-image-04.jpg) | ![After](https://raw.githubusercontent.com/Morteza-Asadi-Shalmaiy/facial-recognition-pipeline-repo/main/assets/results/test-image-04.jpg) |

### Demo Video

| Before | After |
|--------|-------|
| ![Demo](https://raw.githubusercontent.com/Morteza-Asadi-Shalmaiy/facial-recognition-pipeline-repo/refs/heads/main/assets/video-test-01%20(1).gif) | ![Demo](https://raw.githubusercontent.com/Morteza-Asadi-Shalmaiy/facial-recognition-pipeline-repo/refs/heads/main/assets/video-test-01.gif) |
