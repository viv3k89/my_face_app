# facial_recognition.py
import face_recognition
import cv2
import numpy as np
import os
import pickle
import sys

# --- Configuration ---
KNOWN_FACES_DATA_FILE = "known_faces_data.pkl"
KNOWN_FACES_IMAGE_DIR = "known_faces_images"  # Directory to initially load faces from if data file is empty
LFW_DATASET_DIR = "lfw-deepfunneled"  # Path to the extracted LFW dataset (e.g., lfw-deepfunneled)

# Global flag to track if known faces data has changed
known_faces_changed = False

# --- Helper Functions ---

def load_known_faces():
    """
    Loads known face encodings and names from a pickle file.
    If the file doesn't exist or is empty, it scans the LFW_DATASET_DIR
    or KNOWN_FACES_IMAGE_DIR to generate and save the data.
    """
    known_face_encodings = []
    known_face_names = []
    global known_faces_changed  # Declare global to modify it

    # 1. Try loading from the saved pickle file first
    if os.path.exists(KNOWN_FACES_DATA_FILE) and os.path.getsize(KNOWN_FACES_DATA_FILE) > 0:
        try:
            with open(KNOWN_FACES_DATA_FILE, 'rb') as f:
                data = pickle.load(f)
                known_face_encodings = data['encodings']
                known_face_names = data['names']
            print(f"Loaded {len(known_face_names)} known faces from {KNOWN_FACES_DATA_FILE}")
            known_faces_changed = False  # No changes yet
            return known_face_encodings, known_face_names
        except (pickle.UnpicklingError, EOFError) as e:
            print(f"Error loading {KNOWN_FACES_DATA_FILE}: {e}. Attempting to re-scan image directories.")
    else:
        print(f"'{KNOWN_FACES_DATA_FILE}' not found or empty. Scanning image directories for known faces.")

    # 2. If pickle file is empty/corrupted, try loading from LFW dataset
    if os.path.exists(LFW_DATASET_DIR):
        print(f"Scanning LFW dataset from: {LFW_DATASET_DIR}")
        # LFW structure: lfw-deepfunneled/Person_Name/Person_Name_XXXX.jpg
        for person_name in os.listdir(LFW_DATASET_DIR):
            person_dir = os.path.join(LFW_DATASET_DIR, person_name)
            if os.path.isdir(person_dir):  # Ensure it's a directory (a person's folder)
                for filename in os.listdir(person_dir):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(person_dir, filename)
                        try:
                            image = face_recognition.load_image_file(image_path)
                            encodings = face_recognition.face_encodings(image)
                            if encodings:
                                known_face_encodings.append(encodings[0])
                                known_face_names.append(person_name)  # Use folder name as person's name
                            # else: print(f"  No face found in {filename} in LFW. Skipping.") # Too verbose for large datasets
                        except Exception as e:
                            print(f"  Error loading {filename} from LFW: {e}")
        if known_face_encodings:
            print(f"Successfully loaded {len(known_face_names)} faces from LFW dataset.")
            save_known_faces(known_face_encodings, known_face_names)  # Save after initial load
            known_faces_changed = False  # Just saved, so no pending changes
            return known_face_encodings, known_face_names
        else:
            print("No faces found in LFW dataset. Falling back to 'known_faces_images'.")
    else:
        print(f"LFW dataset directory '{LFW_DATASET_DIR}' not found. Falling back to 'known_faces_images'.")


    # 3. Fallback to user's KNOWN_FACES_IMAGE_DIR if LFW not found or empty
    if not os.path.exists(KNOWN_FACES_IMAGE_DIR):
        os.makedirs(KNOWN_FACES_IMAGE_DIR)
        print(f"Created directory: {KNOWN_FACES_IMAGE_DIR}")
        print("Please place images of known individuals in this directory (e.g., 'john_doe.jpg').")
        print("Press 's' during webcam feed to enroll new faces.")
        known_faces_changed = False
        return [], []

    print(f"Scanning user-provided images from: {KNOWN_FACES_IMAGE_DIR}")
    for filename in os.listdir(KNOWN_FACES_IMAGE_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            name = os.path.splitext(filename)[0]
            image_path = os.path.join(KNOWN_FACES_IMAGE_DIR, filename)
            try:
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    known_face_encodings.append(encodings[0])
                    known_face_names.append(name)
                    print(f"  Loaded: {name} from {filename}")
                else:
                    print(f"  No face found in {filename}. Skipping.")
            except Exception as e:
                print(f"  Error loading {filename}: {e}")

    if known_face_encodings:
        save_known_faces(known_face_encodings, known_face_names)  # Save after initial load
        known_faces_changed = False  # Just saved, so no pending changes
    else:
        print("No faces found in 'known_faces_images' directory.")
        known_faces_changed = False

    return known_face_encodings, known_face_names


def save_known_faces(encodings, names):
    """Saves known face encodings and names to a pickle file."""
    with open(KNOWN_FACES_DATA_FILE, 'wb') as f:
        pickle.dump({'encodings': encodings, 'names': names}, f)
    print(f"Saved {len(names)} known faces to {KNOWN_FACES_DATA_FILE}")


def enroll_new_face(frame, known_face_encodings, known_face_names):
    """
    Allows the user to enroll a new face from the current frame.
    Updates in-memory lists and sets a flag for later saving.
    """
    global known_faces_changed  # Declare global to modify it

    # Find all face locations in the full frame
    # We resize the frame for faster detection, but use the original frame for encoding
    small_frame_for_detection = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame_for_detection = cv2.cvtColor(small_frame_for_detection, cv2.COLOR_BGR2RGB)
    face_locations_small = face_recognition.face_locations(rgb_small_frame_for_detection)

    if not face_locations_small:
        print("No face detected in the current frame to enroll. Please try again.")
        return

    # Scale back up the face location to the original frame size
    top, right, bottom, left = face_locations_small[0]
    top *= 4
    right *= 4
    bottom *= 4
    left *= 4
    single_face_location_original_size = [(top, right, bottom, left)]  # face_encodings expects a list of locations

    # Encode the new face using the original full frame and the specific face location
    new_face_encoding = face_recognition.face_encodings(frame, single_face_location_original_size)
    if not new_face_encoding:
        print("Could not encode the detected face. Enrollment failed.")
        return

    # Prompt user for name
    name = input("Enter name for the new face: ").strip()
    if not name:
        print("Enrollment cancelled: Name cannot be empty.")
        return

    # Add to known faces (in-memory)
    if name in known_face_names:
        print(f"Warning: A face with the name '{name}' already exists. Updating its encoding.")
        idx = known_face_names.index(name)
        known_face_encodings[idx] = new_face_encoding[0]
    else:
        known_face_encodings.append(new_face_encoding[0])
        known_face_names.append(name)
        print(f"Enrolled new face: {name}")

    known_faces_changed = True  # Mark that changes have occurred


# --- Main Recognition Loop ---

def run_recognition():
    # Load known faces
    known_face_encodings, known_face_names = load_known_faces()

    # Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("Error: Could not open video stream. Check if webcam is connected and not in use.")
        sys.exit(1)

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    print("\n--- Facial Recognition Started ---")
    print("Press 'q' to quit.")
    print("Press 's' to enroll a new face from the current frame.")

    try:
        while True:
            # Grab a single frame of video
            ret, frame = video_capture.read()
            if not ret:
                print("Failed to grab frame, exiting...")
                break

            # Only process every other frame to save time
            if process_this_frame:
                # Resize frame of video to 1/4 size for faster face recognition processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

                # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                face_names = []
                for face_encoding in face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"

                    # Or instead, use the known face with the smallest distance to the new face
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    if len(face_distances) > 0:  # Ensure there are known faces to compare against
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = known_face_names[best_match_index]

                    face_names.append(name)

            process_this_frame = not process_this_frame

            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            # Display the resulting image
            cv2.imshow('Video', frame)

            # Hit 'q' on the keyboard to quit!
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                enroll_new_face(frame, known_face_encodings, known_face_names)

    finally:
        # Release handle to the webcam
        video_capture.release()
        cv2.destroyAllWindows()
        # Save known faces only once when exiting, if changes were made
        if known_faces_changed:
            print("\nSaving updated known faces data before exiting...")
            save_known_faces(known_face_encodings, known_face_names)
        print("Application closed.")


if __name__ == "__main__":
    run_recognition()
