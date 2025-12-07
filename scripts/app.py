# app.py
# Flask backend for browser-based realtime face recognition + enrollment.
# Place this file in scripts/ and run: python app.py
# Visit: http://127.0.0.1:5000/

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import io
import threading
import base64
import json
import datetime
import numpy as np
import cv2
import face_recognition

app = Flask(__name__, template_folder="templates")
CORS(app)

# Config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "known_faces_data.npz")
NAMES_FILE = os.path.join(BASE_DIR, "known_face_names.json")
IMAGES_DIR = os.path.join(BASE_DIR, "known_faces_images")
MATCH_THRESHOLD = 0.45
DETECTION_MODEL = "hog"
NUM_JITTERS = 1

os.makedirs(IMAGES_DIR, exist_ok=True)

_db_lock = threading.Lock()
_known_encodings = []
_known_names = []


def load_database():
    global _known_encodings, _known_names
    with _db_lock:
        _known_encodings = []
        _known_names = []
        if os.path.exists(DATA_FILE) and os.path.exists(NAMES_FILE):
            try:
                data = np.load(DATA_FILE)
                arr = data.get("encodings", None)
                if arr is not None and arr.size:
                    _known_encodings = [arr[i] for i in range(arr.shape[0])]
                with open(NAMES_FILE, "r", encoding="utf-8") as f:
                    _known_names = json.load(f)
            except Exception as e:
                print("Failed loading DB:", e)
                _known_encodings = []
                _known_names = []


def atomic_save_database(encodings, names):
    # encodings: list of 1D numpy arrays
    tmp_npz = DATA_FILE + ".tmp"
    tmp_json = NAMES_FILE + ".tmp"
    if encodings:
        enc_array = np.stack(encodings)
        np.savez_compressed(tmp_npz, encodings=enc_array)
        os.replace(tmp_npz, DATA_FILE)
    else:
        if os.path.exists(DATA_FILE):
            os.remove(DATA_FILE)
    with open(tmp_json, "w", encoding="utf-8") as f:
        json.dump(names, f, ensure_ascii=False)
    os.replace(tmp_json, NAMES_FILE)


def save_enrollment_image(bgr_img, name):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(c for c in name if c.isalnum() or c in (" ", "_", "-")).rstrip()
    fn = f"{safe}_{ts}.jpg"
    path = os.path.join(IMAGES_DIR, fn)
    cv2.imwrite(path, bgr_img)
    return path


def decode_data_url(data_url):
    # data_url: "data:image/jpeg;base64,...."
    header, encoded = data_url.split(",", 1)
    data = base64.b64decode(encoded)
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
    return img


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/recognize", methods=["POST"])
def recognize():
    payload = request.get_json()
    data_url = payload.get("image")
    if not data_url:
        return jsonify({"error": "no image provided"}), 400

    img_bgr = decode_data_url(data_url)
    if img_bgr is None:
        return jsonify({"error": "failed to decode image"}), 400

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb, model=DETECTION_MODEL)
    encs = face_recognition.face_encodings(rgb, locations, num_jitters=NUM_JITTERS)

    if not encs:
        return jsonify({"result": "no_face"}), 200

    # compare first face (frontend sends single-face crop)
    face_encoding = encs[0]

    with _db_lock:
        if not _known_encodings:
            return jsonify({"result": "unknown", "distance": None, "name": None}), 200
        distances = face_recognition.face_distance(_known_encodings, face_encoding)
        best_idx = int(np.argmin(distances))
        best_dist = float(distances[best_idx])
        if best_dist < MATCH_THRESHOLD:
            return jsonify({"result": "match", "name": _known_names[best_idx], "distance": best_dist}), 200
        else:
            return jsonify({"result": "unknown", "distance": best_dist, "name": None}), 200


@app.route("/enroll", methods=["POST"])
def enroll():
    payload = request.get_json()
    data_url = payload.get("image")
    name = payload.get("name", "").strip()
    if not data_url or not name:
        return jsonify({"error": "image and name required"}), 400

    img_bgr = decode_data_url(data_url)
    if img_bgr is None:
        return jsonify({"error": "failed to decode image"}), 400

    # compute encoding
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    locs = face_recognition.face_locations(rgb, model=DETECTION_MODEL)
    encs = face_recognition.face_encodings(rgb, locs, num_jitters=NUM_JITTERS)
    if not encs:
        return jsonify({"error": "no face found in image"}), 400

    enc = encs[0]

    # Save image and append to DB
    with _db_lock:
        # persist image
        save_enrollment_image(img_bgr, name)
        # update in-memory
        _known_encodings.append(enc)
        _known_names.append(name)
        # atomic save
        atomic_save_database(_known_encodings, _known_names)

    return jsonify({"result": "enrolled", "name": name}), 200


if __name__ == "__main__":
    load_database()
    print("Starting server on http://127.0.0.1:5000/ - press Ctrl+C to stop")
    app.run(host="127.0.0.1", port=5000, debug=False)
    print("Server stopped.")