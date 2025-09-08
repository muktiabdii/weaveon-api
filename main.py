from flask import Flask, request, jsonify
import cv2
import os
from fer import FER
from werkzeug.utils import secure_filename

# Konfigurasi dasar
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'temp_video'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Inisialisasi detector FER
detector = FER()

# Bobot untuk masing-masing emosi
emotion_weights = {
    "angry": -3,
    "disgust": -3,
    "fear": -2,
    "happy": 2,
    "sad": -2,
    "surprise": 1,
    "neutral": 0
}

# Fungsi bantu untuk ekstrak frame dari video
def extract_frames(video_path, max_frames=60):
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

# Fungsi untuk menghitung distribusi dan skor emosi
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
        return {
            "score": 0,
            "label": "Tidak terdeteksi",
            "distribution": {},
        }

    score = total_weight / frame_count

    if score > 1.2:
        label = "Sangat senang"
    elif score >= 0.5:
        label = "Cukup senang" # senang
    elif score > -0.5:
        label = "Netral" # hapus
    elif score >= -1.2:
        label = "Kurang senang" # tidak senang
    else:
        label = "Sangat tidak senang"

    # Hitung persentase distribusi
    distribution = {
        emo: round((count / frame_count) * 100, 2)
        for emo, count in emotion_counts.items()
        if count > 0
    }

    return {
        "score": round(score, 2),
        "label": label,
        "distribution": distribution,
    }

# Endpoint API
@app.route('/analyze', methods=['POST'])
def analyze():
    if 'video' not in request.files:
        return jsonify({"error": "No video part in the request"}), 400

    video = request.files['video']
    if video.filename == '':
        return jsonify({"error": "No selected video"}), 400

    filename = secure_filename(video.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video.save(video_path)

    try:
        frames = extract_frames(video_path)
        result = analyze_emotions(frames)
    finally:
        os.remove(video_path)  # Hapus video setelah analisis selesai

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
