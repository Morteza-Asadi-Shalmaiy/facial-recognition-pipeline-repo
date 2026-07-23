# -*- coding: utf-8 -*-
"""Face Recognition Module

Part of a multi-module CV pipeline (liveness/anti-spoofing, face recognition,
super-resolution, person Re-ID) running entirely locally on Google Colab,
with no external API dependencies.

Originally developed as a Colab notebook:
    18/06/2026-mu-Base-Facial-recognation.ipynb
"""

# ---- Iranian date and time utility ----
# !pip -q install jdatetime

import jdatetime
from datetime import datetime
from zoneinfo import ZoneInfo

def get_iran_datetime_stamp():
    # Get the current date and time in Iran
    iran_time_now = datetime.now(ZoneInfo("Asia/Tehran"))
    iran_time_formatted = iran_time_now.strftime("%H:%M:%S")

    # Get the current Jalali date
    iran_jalali_now = jdatetime.date.today()

    # Explicitly cast to string to ensure both time and date are returned
    return f"{iran_time_formatted} {str(iran_jalali_now)}"


# ---- 1. Installing requirements & importing them ----
# !pip -q install insightface onnxruntime opencv-python-headless

import os
import cv2
import numpy as np
from pathlib import Path
import insightface
from insightface.app import FaceAnalysis
from datetime import datetime


# ---- 2. Initializing the InsightFace model, saved faces, embedding them ----

app = FaceAnalysis(
    name='buffalo_l',
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

app.prepare(
    ctx_id=0,
    det_size=(640, 640))


def build_face_db(faces_dir: str, app: FaceAnalysis):
    known_embeddings = []
    known_names = []

    for img_path in Path(faces_dir).iterdir():

        # checking for right format
        if img_path.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: could not read {img_path.name}")
            continue

        faces = app.get(img)
        if not faces:
            print(f"Warning: no face detected in {img_path.name}, skipping.")
            continue

        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
        known_embeddings.append(face.normed_embedding)
        known_names.append(img_path.stem)

    print(f"Loaded {len(known_names)} known faces: {known_names}")
    return known_embeddings, known_names


# Example: mounting Google Drive and loading reference faces (Colab-specific)
# from google.colab import drive
# drive.mount('/content/drive')
# !cp -r /content/drive/MyDrive/facial-recognition-ai/faces /content
#
# FACES_DIR = "/content/faces"
# known_embeddings, known_names = build_face_db(FACES_DIR, app)


# ---- 3. Processing video and image functions ----

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def process_image(input_path, output_path, known_embeddings, known_names, app, threshold=0.4):
    img = cv2.imread(input_path)
    faces = app.get(img)
    timestamp = get_iran_datetime_stamp()
    results = []

    for face in faces:
        box = face.bbox.astype(int)
        name = "Unknown"
        similarity_percent = 0.0
        color = (0, 0, 255)

        if known_embeddings:
            sims = [cosine_similarity(face.normed_embedding, k) for k in known_embeddings]
            best_idx = np.argmax(sims)
            best_score = sims[best_idx]
            similarity_percent = float(best_score * 100)
            name = known_names[best_idx]
            color = (0, 255, 0)

        results.append({
            "name": name,
            "similarity_percent": f"{similarity_percent:.2f}%",
            "timestamp": timestamp
        })

        label = f"{name} ({similarity_percent:.0f}%) {timestamp}"
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
        cv2.putText(img, label, (box[0], box[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imwrite(output_path, img)

    return results


def process_video(
    input_path: str,
    output_path: str,
    known_embeddings: list,
    known_names: list,
    app: FaceAnalysis,
    threshold: float = 0.4,
):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    video_results = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = app.get(frame)
        timestamp = get_iran_datetime_stamp()

        for face in faces:
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            name = "Unknown"
            similarity_percent = 0.0
            color = (0, 0, 255)

            if known_embeddings:
                sims = [cosine_similarity(face.normed_embedding, e) for e in known_embeddings]
                best_idx = np.argmax(sims)
                best_score = sims[best_idx]
                similarity_percent = float(best_score * 100)
                name = known_names[best_idx]
                color = (0, 255, 0)

            video_results.append({
                "frame": frame_idx,
                "name": name,
                "similarity_percent": f"{similarity_percent:.2f}%",
                "timestamp": timestamp
            })

            label = f"{name} ({similarity_percent:.0f}%) {timestamp}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(frame, (x1, y2 - 28), (x2, y2), color, cv2.FILLED)
            cv2.putText(frame, label, (x1 + 4, y2 - 8),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

        out.write(frame)
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx}/{total} frames")

    cap.release()
    out.release()
    print(f"Done. Output saved to: {output_path}")
    return video_results


# ---- 4. Example usage ----
if __name__ == "__main__":
    # Single frame example
    # results = process_image(
    #     "/content/Screenshot.jpg",
    #     "/content/output-Screenshot.jpg",
    #     known_embeddings, known_names, app)

    # Video example
    # video_results = process_video(
    #     "/content/input_video.mp4",
    #     "/content/output_video.mp4",
    #     known_embeddings, known_names, app)
    pass
