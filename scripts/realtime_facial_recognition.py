# realtime_facial_recognition.py
# Clean version: NO LFW, ONLY user-enrolled faces.

import cv2
import face_recognition
import numpy as np
import os
import json
import datetime

# -----------------------------------------
# CONFIG
# -----------------------------------------
MATCH_THRESHOLD = 0.45          # Lower = stricter. Adjust 0.45–0.55 if needed.
DATA_FILE = "known_faces_data.npz"
NAMES_FILE = "known_face_names.json"
IMAGES_DIR = "known_faces_images"  # folder to save enrolled face photos
DETECTION_MODEL = "hog"            # hog = CPU, good speed

# Resolve absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, DATA_FILE)
NAMES_PATH = os.path.join(BASE_DIR, NAMES_FILE)
IMAGES_PATH = os.path.join(BASE_DIR, IMAGES_DIR)

os.makedirs(IMAGES_PATH, exist_ok=True)


# -----------------------------------------
# LOAD / SAVE FUNCTIONS
# -----------------------------------------
def load_database():
    """Load encodings + names from disk."""
    if not os.path.exists(DATA_PATH) or not os.path.exists(NAMES_PATH):
        return [], []

    try:
        npz = np.load(DATA_PATH)
        encodings = [npz["encodings"][i] for i in range(npz["encodings"].shape[0])]
        names = json.load(open(NAMES_PATH, "r", encoding="utf-8"))
        return encodings, names
    except:
        return [], []


def save_database(encodings, names):
    """Save encodings + names atomically."""
    arr = np.stack(encodings) if len(encodings) > 0 else np.empty((0, 128))
    np.savez_compressed(DATA_PATH, encodings=arr)

    with open(NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(names, f, ensure_ascii=False)

    print("[INFO] Database updated.")


# -----------------------------------------
# ENROLL NEW FACE
# -----------------------------------------
def enroll_face(frame, location, known_encodings, known_names):
    top, right, bottom, left = location

    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    encoding = face_recognition.face_encodings(rgb_frame, [location])
    if not encoding:
        print("[ERROR] Could not encode face.")
        return

    encoding = encoding[0]

    # Ask user for name
    name = input("Enter name for this person: ").strip()
    if not name:
        print("[WARN] Empty name. Enrollment cancelled.")
        return

    # SAVE FACE IMAGE
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_{timestamp}.jpg"
    filepath = os.path.join(IMAGES_PATH, filename)

    face_crop = frame[top:bottom, left:right]
    cv2.imwrite(filepath, face_crop)

    # UPDATE DATABASE
    known_encodings.append(encoding)
    known_names.append(name)
    save_database(known_encodings, known_names)

    print(f"[INFO] Enrolled new face: {name}")


# -----------------------------------------
# MAIN LOOP
# -----------------------------------------
def main():
    known_encodings, known_names = load_database()
    print(f"[INFO] Loaded {len(known_names)} known faces.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    print("[INFO] Press 's' to enroll face. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Frame capture failed.")
            break

        # Resize small for detection (faster)
        small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        locations = face_recognition.face_locations(rgb_small, model=DETECTION_MODEL)
        encodings = face_recognition.face_encodings(rgb_small, locations)

        face_names = []

        for encoding in encodings:
            if len(known_encodings) == 0:
                face_names.append("Unknown")
                continue

            distances = face_recognition.face_distance(known_encodings, encoding)
            best_index = np.argmin(distances)
            best_distance = distances[best_index]

            if best_distance < MATCH_THRESHOLD:
                name = known_names[best_index]
            else:
                name = "Unknown"

            face_names.append(name)

        # Draw results
        for (top, right, bottom, left), name in zip(locations, face_names):
            # Scale back up
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw box + label
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), 
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

        cv2.imshow("Real-Time Face Recognition", frame)

        key = cv2.waitKey(1) & 0xFF

        # Quit program
        if key == ord('q'):
            break

        # Enroll face (largest detected)
        if key == ord('s'):
            if len(locations) == 0:
                print("[WARN] No face visible.")
            else:
                # choose largest face
                areas = [(b - t) * (r - l) for (t, r, b, l) in locations]
                idx = np.argmax(areas)

                # scale coordinates back up
                (t, r, b, l) = locations[idx]
                t, r, b, l = t*4, r*4, b*4, l*4

                print("[INFO] Enrolling face...")
                enroll_face(frame, (t, r, b, l), known_encodings, known_names)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
