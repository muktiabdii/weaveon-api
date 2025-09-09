import os
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fer import FER
from tempfile import NamedTemporaryFile

# Setup FastAPI
app = FastAPI(title="Emotion Analyzer API")

UPLOAD_FOLDER = "temp_video"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Inisialisasi detector FER
detector = FER(mtcnn=False)  # lebih ringan tanpa MTCNN

# Bobot emosi
emotion_weights = {
    "angry": -3,
    "disgust": -3,
    "fear": -2,
    "happy": 2,
    "sad": -2,
    "surprise": 1,
    "neutral": 0
}

# Fungsi ekstrak frame dari video
def extract_frames(video_path, max_frames=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total_frames // max_frames)
    count = 0

    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            frames.append(frame)
        count += 1

    cap.release()
    return frames

# Fungsi analisis emosi
def analyze_emotions(frames):
    emotion_counts = {emo: 0 for emo in emotion_weights.keys()}
    total_weight = 0
    frame_count = 0

    for frame in frames:
        result = detector.top_emotion(frame)
        if result:
            emotion, _ = result
            if emotion in emotion_counts:
                emotion_counts[emotion] += 1
                total_weight += emotion_weights[emotion]
                frame_count += 1

    if frame_count == 0:
        return {"score": 0, "label": "Tidak terdeteksi", "distribution": {}}

    score = total_weight / frame_count

    if score > 1.2:
        label = "Sangat senang"
    elif score >= 0.5:
        label = "Cukup senang"
    elif score > -0.5:
        label = "Netral"
    elif score >= -1.2:
        label = "Kurang senang"
    else:
        label = "Sangat tidak senang"

    distribution = {
        emo: round((count / frame_count) * 100, 2)
        for emo, count in emotion_counts.items() if count > 0
    }

    return {"score": round(score, 2), "label": label, "distribution": distribution}

# Endpoint FastAPI
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No selected video")

    # Simpan sementara video ke temp file
    with NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        frames = extract_frames(tmp_path)
        result = analyze_emotions(frames)
    finally:
        os.remove(tmp_path)  # hapus file setelah selesai

    return JSONResponse(content=result)
