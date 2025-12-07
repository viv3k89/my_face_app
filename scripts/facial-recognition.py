# face_recognition_test.py
# Static-image test: compares one unknown image inside known_faces_images/ to the known faces in that folder.
# Usage: put known face images and an unknown image (unknown_image.jpg) in scripts/known_faces_images/
#        then run: python face_recognition_test.py

import os
import face_recognition
import numpy as np

KNOWN_FACES_DIR = "known_faces_images"
UNKNOWN_IMAGE = "unknown_image.jpg"
MATCH_THRESHOLD = 0.45  # keep consistent with realtime script

def main():
    base = os.path.dirname(os.path.abspath(__file__))
    known_dir = os.path.join(base, KNOWN_FACES_DIR)
    unknown_path = os.path.join(known_dir, UNKNOWN_IMAGE)

    if not os.path.isdir(known_dir):
        print(f"Directory not found: {known_dir}")
        print("Create it and add known faces and an unknown_image.jpg to test.")
        return
    if not os.path.exists(unknown_path):
        print(f"Unknown image not found at {unknown_path}")
        return

    known_encodings = []
    known_names = []
    for fname in os.listdir(known_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        if fname == UNKNOWN_IMAGE:
            continue
        name = os.path.splitext(fname)[0]
        path = os.path.join(known_dir, fname)
        try:
            img = face_recognition.load_image_file(path)
            encs = face_recognition.face_encodings(img)
            if encs:
                known_encodings.append(encs[0])
                known_names.append(name)
                print(f"Loaded known face: {name}")
            else:
                print(f"No face found in {fname}; skipped.")
        except Exception as e:
            print(f"Error loading {fname}: {e}")

    if not known_encodings:
        print("No known face encodings loaded. Add images to the folder.")
        return

    unknown_img = face_recognition.load_image_file(unknown_path)
    face_locations = face_recognition.face_locations(unknown_img)
    face_encodings = face_recognition.face_encodings(unknown_img, face_locations)

    print(f"Found {len(face_encodings)} face(s) in the unknown image.")
    for (top, right, bottom, left), fe in zip(face_locations, face_encodings):
        distances = face_recognition.face_distance(known_encodings, fe)
        best_idx = int(np.argmin(distances))
        best_distance = float(distances[best_idx])
        if best_distance < MATCH_THRESHOLD:
            print(f"Match: {known_names[best_idx]} (distance={best_distance:.3f})")
        else:
            print(f"No good match (closest distance={best_distance:.3f})")

if __name__ == "__main__":
    main()
