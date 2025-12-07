# camera_test.py
# Robust webcam opener and simple camera display test.
# Usage: python camera_test.py
import cv2
import sys
import time

def open_camera_try(index=0):
    """
    Try multiple backends and indices to open a webcam reliably.
    Returns an opened cv2.VideoCapture or None.
    """
    backends = []
    try:
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    except Exception:
        backends = [cv2.CAP_ANY]

    for backend in backends:
        try:
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                return cap
            cap.release()
        except Exception:
            # some backends may not be supported on the platform
            pass

    # Fallback: try other likely indices
    for idx in (1, 2, 3):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            return cap
        cap.release()

    return None

def main():
    print("Attempting to open webcam...")
    cap = open_camera_try(0)
    if cap is None:
        print("Error: Could not open webcam. Ensure no other app uses it and check privacy settings.")
        sys.exit(1)

    print("Webcam opened successfully. Press 'q' to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame. Exiting...")
                break

            cv2.imshow("Webcam Test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # small sleep to keep CPU usage reasonable
            time.sleep(0.001)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam test finished and closed properly.")

if __name__ == "__main__":
    main()
