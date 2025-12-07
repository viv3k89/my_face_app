# app.py
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os, io, base64, json, time
from PIL import Image
import numpy as np
import face_recognition

# ----------------- Configuration -----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # scripts/
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
KNOWN_DIR = os.path.join(BASE_DIR, "known_faces_images")
NPZ_FILE = os.path.join(BASE_DIR, "known_faces_data.npz")
NAMES_FILE = os.path.join(BASE_DIR, "known_face_names.json")

# Matching threshold (lower = stricter)
MATCH_THRESHOLD = 0.45

app = Flask(__name__, template_folder=TEMPLATES_DIR)
CORS(app)

# In-memory cache (kept minimal). We reload from disk on demand.
_known_encodings = []
_known_names = []

# ----------------- Utility functions -----------------
def ensure_dirs():
    os.makedirs(KNOWN_DIR, exist_ok=True)

def save_db(encodings, names):
    # encodings: list of 128-d numpy arrays
    if encodings:
        arr = np.vstack([np.asarray(e) for e in encodings])
    else:
        arr = np.zeros((0,128), dtype=np.float64)
    np.savez_compressed(NPZ_FILE, encodings=arr)
    with open(NAMES_FILE, "w", encoding="utf-8") as f:
        json.dump(names, f, ensure_ascii=False)
    print(f"[DB] Saved {len(names)} names, enc shape {arr.shape} to disk.")

def load_db():
    """Load encodings and names from NPZ+JSON files into _known_encodings/_known_names."""
    global _known_encodings, _known_names
    _known_encodings = []
    _known_names = []
    try:
        if os.path.exists(NPZ_FILE):
            d = np.load(NPZ_FILE, allow_pickle=True)
            arr = d.get("encodings")
            if arr is not None and arr.size:
                # arr shape (N, 128)
                _known_encodings = [arr[i] for i in range(arr.shape[0])]
        if os.path.exists(NAMES_FILE):
            with open(NAMES_FILE, "r", encoding="utf-8") as f:
                _known_names = json.load(f)
    except Exception as e:
        print("[DB] load error:", e)
        _known_encodings, _known_names = [], []
    print(f"[DB] Loaded {len(_known_names)} names, {len(_known_encodings)} encodings from disk.")

def decode_dataurl(data_url):
    """Decode data:url base64 image -> PIL Image"""
    if not data_url:
        return None
    header, b64 = data_url.split(",", 1) if "," in data_url else (None, data_url)
    img_bytes = base64.b64decode(b64)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")

def pil_to_np(img):
    """PIL Image -> RGB numpy array expected by face_recognition (H,W,3)"""
    return np.array(img)

# ----------------- Routes -----------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/recognize", methods=["POST"])
def recognize():
    # Always reload DB to avoid stale in-memory state after restarts or external edits.
    load_db()

    data = request.get_json() or {}
    data_url = data.get("image")
    if not data_url:
        return jsonify({"result":"error", "message":"no image provided"}), 400

    pil = decode_dataurl(data_url)
    if pil is None:
        return jsonify({"result":"error", "message":"invalid image"}), 400

    img = pil_to_np(pil)
    # detect in full-size; you may downscale on client for speed
    face_locations = face_recognition.face_locations(img)
    if not face_locations:
        return jsonify({"result":"no_face"})

    face_encodings = face_recognition.face_encodings(img, face_locations)
    # We'll return best match for the first face only
    enc = face_encodings[0]
    if not _known_encodings:
        return jsonify({"result":"unknown", "distance": None})

    distances = face_recognition.face_distance(_known_encodings, enc)
    best_idx = int(np.argmin(distances))
    best_distance = float(distances[best_idx])

    if best_distance < MATCH_THRESHOLD:
        name = _known_names[best_idx] if best_idx < len(_known_names) else "Unknown"
        return jsonify({"result":"match", "name": name, "distance": best_distance})
    else:
        return jsonify({"result":"unknown", "distance": best_distance})

@app.route("/enroll", methods=["POST"])
def enroll():
    # Accepts { image: dataURL, name: "Alice" }
    data = request.get_json() or {}
    data_url = data.get("image")
    name = data.get("name", "").strip()
    if not data_url or not name:
        return jsonify({"result":"error", "message":"image and name required"}), 400

    pil = decode_dataurl(data_url)
    if pil is None:
        return jsonify({"result":"error", "message":"invalid image"}), 400

    img = pil_to_np(pil)
    # detect faces
    face_locations = face_recognition.face_locations(img)
    if not face_locations:
        return jsonify({"result":"error", "message":"no face detected"}), 400

    # choose first face for enrollment
    face_encoding = face_recognition.face_encodings(img, [face_locations[0]])
    if not face_encoding:
        return jsonify({"result":"error", "message":"could not encode face"}), 500
    enc = face_encoding[0]

    # ensure dirs & load DB
    ensure_dirs()
    load_db()

    # Save face image file to known_faces_images with timestamp
    safe_name = "".join(c for c in name if c.isalnum() or c in (" ", "-", "_")).strip().replace(" ", "_")
    fname = f"{safe_name}_{int(time.time())}.jpg"
    fpath = os.path.join(KNOWN_DIR, fname)
    pil.save(fpath, format="JPEG", quality=90)

    # Append to in-memory lists, then persist (NPZ+JSON)
    _known_encodings.append(enc)
    _known_names.append(name)
    save_db(_known_encodings, _known_names)

    return jsonify({"result":"enrolled", "name": name})

# serve known images (optional)
@app.route("/known_faces/<path:filename>")
def known_faces_file(filename):
    return send_from_directory(KNOWN_DIR, filename)

# ----------------- Startup -----------------
if __name__ == "__main__":
    ensure_dirs()
    load_db()
    print("Starting Flask app on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
