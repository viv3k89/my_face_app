# camera_test.py
import cv2
import sys

print("Attempting to open webcam...")

# IMPORTANT for Windows webcam stability:
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    print("Possible causes:")
    print("  - Another application is using the camera (Zoom, Teams, WhatsApp, browser, etc.)")
    print("  - Wrong camera index (try index 1 or 2)")
    print("  - Webcam disabled in Windows privacy settings")
    print("\nFixes:")
    print("  1) Close other camera apps")
    print("  2) Try: cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)")
    print("  3) Check: Settings > Privacy & Security > Camera")
    sys.exit(1)

print("Webcam opened successfully! Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    cv2.imshow('Webcam Test', frame)

    # QUIT IF USER PRESSES q (on the webcam window)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Webcam test finished and closed properly.")
