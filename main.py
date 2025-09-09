import os
import cv2
import numpy as np
import logging
import asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fer import FER
from tempfile import NamedTemporaryFile

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Setup FastAPI
app = FastAPI(title="Emotion Analyzer API")

UPLOAD_FOLDER = "temp_video"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Inisialisasi detector FER
detector = FER(mtcnn=True)

# Bobot emosi
emotion_weights = {
    "angry": -1.0,
    "disgust": -0.8,
    "fear": -0.6,
    "happy": 1.0,
    "sad": -0.7,
    "surprise": 0.5,
    "neutral": 0.0
}

def preprocess_frame(frame):
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (160, 160), interpolation=cv2.INTER_AREA)
        return frame_resized  # Remove normalization to keep uint8
    except Exception as e:
        logger.error(f"Error preprocessing frame: {str(e)}")
        return frame

def extract_frames(video_path, max_frames=10):
    logger.debug(f"Extracting frames from {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video file: {video_path}")
        raise ValueError("Cannot open video file")
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total_frames // max_frames)
    count = 0

    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            frames.append(preprocess_frame(frame))
        count += 1

    cap.release()
    logger.debug(f"Extracted {len(frames)} frames from {total_frames} total frames")
    return frames

def analyze_emotions(frames, temporal_weighting=True):
    logger.debug(f"Analyzing emotions for {len(frames)} frames")
    emotion_scores = []
    emotion_counts = {emo: 0 for emo in emotion_weights.keys()}
    frame_count = 0
    total_frames = len(frames)

    for i, frame in enumerate(frames):
        logger.debug(f"Processing frame {i}")
        try:
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
                logger.debug(f"Frame {i} analyzed successfully, score={frame_score}")
            else:
                emotion_scores.append(0.0)
                logger.debug(f"No emotions detected in frame {i}")
        except Exception as e:
            logger.error(f"Error analyzing frame {i}: {str(e)}")
            emotion_scores.append(0.0)

    if frame_count == 0:
        logger.warning("No emotions detected in any frame")
        return {
            "score": 0.0,
            "score_std": 0.0,
            "label": "Tidak terdeteksi",
            "distribution": {},
            "confidence": 0.0
        }

    confidence = frame_count / total_frames if total_frames > 0 else 0.0
    if confidence < 0.3:
        logger.warning(f"Low confidence ({confidence*100:.2f}%), result may be unreliable")
        return {
            "score": 0.0,
            "score_std": 0.0,
            "label": "Deteksi tidak cukup",
            "distribution": distribution,
            "confidence": round(confidence * 100, 2)
        }

    score = np.mean(emotion_scores)
    score_std = np.std(emotion_scores) if len(emotion_scores) > 1 else 0.0
    score = max(min(score, 1.0), -1.0)

    if score > 0.6:
        label = "Sangat senang"
    elif score >= 0.2:
        label = "Cukup senang"
    elif score >= -0.1:
        label = "Netral"
    elif score >= -0.4:
        label = "Kurang senang"
    else:
        label = "Sangat tidak senang"

    distribution = {
        emo: round((count / frame_count) * 100, 2)
        for emo, count in emotion_counts.items() if count > 0
    }

    logger.info(f"Analysis result: score={score}, confidence={confidence*100:.2f}%, distribution={distribution}")
    return {
        "score": round(score, 2),
        "score_std": round(score_std, 2),
        "label": label,
        "distribution": distribution,
        "confidence": round(confidence * 100, 2)
    }

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    logger.info(f"Received file: {file.filename}, size: {file.size} bytes")
    if not file.filename:
        raise HTTPException(status_code=400, detail="No selected video")
    if not file.filename.lower().endswith(".mp4"):
        raise HTTPException(status_code=400, detail="File harus berformat MP4")
    if file.size > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File terlalu besar, maksimum 10 MB")

    with NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        try:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
            logger.debug(f"Temporary file saved at {tmp_path}")
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            raise HTTPException(status_code=500, detail="Error saving video file")

    try:
        async with asyncio.timeout(30):
            frames = extract_frames(tmp_path)
            result = analyze_emotions(frames, temporal_weighting=True)
            logger.info("Processing completed successfully")
    except asyncio.TimeoutError:
        logger.error("Processing timed out after 30 seconds")
        raise HTTPException(status_code=504, detail="Processing timed out")
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
    finally:
        try:
            os.remove(tmp_path)
            logger.debug(f"Temporary file deleted: {tmp_path}")
        except Exception as e:
            logger.error(f"Error deleting temp file: {str(e)}")

    return JSONResponse(content=result)