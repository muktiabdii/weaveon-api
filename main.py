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

# Inisialisasi detector FER (lebih ringan tanpa MTCNN)
detector = FER(mtcnn=False)

# Bobot emosi (skala -1 sampai 1)
emotion_weights = {
    "angry": -1.0,
    "disgust": -0.8,
    "fear": -0.6,
    "happy": 1.0,
    "sad": -0.7,
    "surprise": 0.5,
    "neutral": 0.0
}

# Ekstraksi frame dari video
def extract_frames(video_path, max_frames=60):
    frames = []
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total == 0:
        return frames

    step = max(1, total // max_frames)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % step == 0:
            frame = cv2.resize(frame, (640, 480))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        count += 1

    cap.release()
    return frames

# Analisis emosi
def analyze_emotions(frames, temporal_weighting=True):
    emotion_scores = []
    emotion_counts = {emo: 0 for emo in emotion_weights.keys()}
    frame_count = 0
    total_frames = len(frames)

    for i, frame in enumerate(frames):
        print(f"Processing frame {i+1}/{total_frames}")  # log debug

        # pastikan frame RGB 3 channel
        if len(frame.shape) == 2:  # grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:  # RGBA
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

        result = detector.detect_emotions(frame)
        if result:
            emotions = result[0]["emotions"]

            frame_score = 0
            for emo, prob in emotions.items():
                if emo in emotion_weights:
                    frame_score += prob * emotion_weights[emo]
                    if prob > 0.1:
                        emotion_counts[emo] += 1

            if temporal_weighting and total_frames > 0:
                weight = 1.0 + 0.5 * (i / total_frames)
                frame_score *= weight

            emotion_scores.append(frame_score)
            frame_count += 1

    if frame_count == 0:
        return {
            "score": 0.0,
            "label": "Tidak terdeteksi",
            "distribution": {}
        }

    score = np.mean(emotion_scores)
    score = max(min(score, 1.0), -1.0)  # clamp -1..1

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
        "label": label,
        "distribution": distribution
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
        print(f"Total frames extracted: {len(frames)}")  # log debug
        result = analyze_emotions(frames)
    finally:
        os.remove(tmp_path)

    return JSONResponse(content=result)
