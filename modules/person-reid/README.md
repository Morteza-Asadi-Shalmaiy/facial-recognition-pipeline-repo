# Person Re-Identification (Re-ID) Module

**Status:** In progress — add notebook/code and update this README once finalized.

## Planned Overview
- TorchReID (OSNet) + YOLOv8 for person detection and re-identification
- Fallback stage when face recognition and super-resolution can't resolve an identity
- Additional features explored: clothing color detection, height estimation via MediaPipe Pose, multi-camera tracking timelines, auto-generated case file summaries

## Datasets Reviewed
- Market-1501 (recommended starting point)
- MSMT17
- WildTrack

## To Do Before Presenting
- [ ] Add final code/notebook
- [ ] Add Re-ID accuracy metrics (mAP, Rank-1) on chosen dataset
- [ ] Document multi-camera tracking approach
