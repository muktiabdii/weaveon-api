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

# Bobot emosi yang lebih seimbang (disesuaikan berdasarkan skala -1 hingga 1 untuk normalisasi)
emotion_weights = {
    "angry": -1.0,
    "disgust": -0.8,
    "fear": -0.6,
    "happy": 1.0,
    "sad": -0.7,
    "surprise": 0.5,
    "neutral": 0.0
}

def analyze_emotions(frames, temporal_weighting=True):
    emotion_scores = []  # Menyimpan vektor skor per frame
    emotion_counts = {emo: 0 for emo in emotion_weights.keys()}
    frame_count = 0
    total_frames = len(frames)

    for i, frame in enumerate(frames):
        # Gunakan FER untuk mendapatkan distribusi probabilitas emosi
        result = detector.detect_emotions(frame)
        if result and len(result) > 0:
            # Ambil probabilitas emosi dari deteksi pertama (asumsi satu wajah per frame)
            emotions = result[0]["emotions"]
            
            # Hitung skor frame sebagai kombinasi probabilitas * bobot
            frame_score = 0
            for emo, prob in emotions.items():
                if emo in emotion_weights:
                    frame_score += prob * emotion_weights[emo]
                    if prob > 0.1:  # Hitung emosi dengan probabilitas signifikan
                        emotion_counts[emo] += 1
            
            # Terapkan bobot temporal (opsional)
            if temporal_weighting:
                # Berikan bobot lebih pada frame di 50% terakhir video
                weight = 1.0 + 0.5 * (i / total_frames) if total_frames > 0 else 1.0
                frame_score *= weight
            
            emotion_scores.append(frame_score)
            frame_count += 1
        else:
            # Jika deteksi gagal, gunakan skor netral (0) untuk menghindari bias
            emotion_scores.append(0.0)

    if frame_count == 0:
        return {
            "score": 0.0,
            "score_std": 0.0,
            "label": "Tidak terdeteksi",
            "distribution": {},
            "confidence": 0.0
        }

    # Hitung statistik
    score = np.mean(emotion_scores)  # Rata-rata skor
    score_std = np.std(emotion_scores) if len(emotion_scores) > 1 else 0.0  # Standar deviasi
    confidence = frame_count / total_frames if total_frames > 0 else 0.0  # Proporsi frame terdeteksi

    # Normalisasi skor ke [-1, 1] jika diperlukan
    score = max(min(score, 1.0), -1.0)

    # Tentukan label berdasarkan skor
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

    # Hitung distribusi emosi
    distribution = {
        emo: round((count / frame_count) * 100, 2)
        for emo, count in emotion_counts.items() if count > 0
    }

    return {
        "score": round(score, 2),
        "score_std": round(score_std, 2),  # Tambahan: variabilitas skor
        "label": label,
        "distribution": distribution,
        "confidence": round(confidence * 100, 2)  # Persentase frame terdeteksi
    }

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
