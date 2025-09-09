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
detector = FER(mtcnn=True)  

# Bobot emosi yang lebih seimbang (skala -1 sampai 1)
emotion_weights = {
    "angry": -1.0,
    "disgust": -0.8,
    "fear": -0.6,
    "happy": 1.0,
    "sad": -0.7,
    "surprise": 0.5,
    "neutral": 0.0
}

# Fungsi untuk ekstraksi frame dari video
def extract_frames(video_path, max_frames=100):
    frames = []
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Batasin jumlah frame biar tidak terlalu berat
    step = max(1, total // max_frames)

    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % step == 0:
            # Resize biar lebih ringan untuk analisis
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
        count += 1

    cap.release()
    return frames

# Fungsi untuk analisis emosi
def analyze_emotions(frames, temporal_weighting=True):
    emotion_scores = []
    emotion_counts = {emo: 0 for emo in emotion_weights.keys()}
    frame_count = 0
    total_frames = len(frames)

    for i, frame in enumerate(frames):
        result = detector.detect_emotions(frame)
        if result and len(result) > 0:
            emotions = result[0]["emotions"]

            frame_score = 0
            for emo, prob in emotions.items():
                if emo in emotion_weights:
                    frame_score += prob * emotion_weights[emo]
                    if prob > 0.1:
                        emotion_counts[emo] += 1

            if temporal_weighting:
                weight = 1.0 + 0.5 * (i / total_frames) if total_frames > 0 else 1.0
                frame_score *= weight

            emotion_scores.append(frame_score)
            frame_count += 1
        else:
            emotion_scores.append(0.0)

    if frame_count == 0:
        return {
            "score": 0.0,
            "score_std": 0.0,
            "label": "Tidak terdeteksi",
            "distribution": {},
            "confidence": 0.0
        }

    score = np.mean(emotion_scores)
    score_std = np.std(emotion_scores) if len(emotion_scores) > 1 else 0.0
    confidence = frame_count / total_frames if total_frames > 0 else 0.0

    score = max(min(score, 1.0), -1.0)

    if score > 0.6:
        label = "Sangat senang"
    elif score >= 0.2:
        label = "Cukup senang"
    elif score > -0.2:
        label = "Netral"
    elif score >= -0.6:
        label = "Kurang senang"
    else:
        label = "Sangat tidak senang"

    distribution = {
        emo: round((count / frame_count) * 100, 2)
        for emo, count in emotion_counts.items() if count > 0
    }

    return {
        "score": round(score, 2),
        "score_std": round(score_std, 2),
        "label": label,
        "distribution": distribution,
        "confidence": round(confidence * 100, 2)
    }

# Endpoint FastAPI
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No selected video")

    with NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        frames = extract_frames(tmp_path)
        result = analyze_emotions(frames)
    finally:
        os.remove(tmp_path)

    return JSONResponse(content=result)
