import cv2
import mediapipe as mp
import numpy as np

# Mediapipe setup
mp_hands = mp.solutions.hands

# Webcam
cap = cv2.VideoCapture(0)

# White canvas to draw
canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255

# Track previous finger point
prev_x, prev_y = None, None

def is_palm_open(landmarks):
    """
    Check if palm is open:
    We say palm is open if fingers 5–8 (index to pinky tips) are extended above knuckles.
    """
    finger_tips = [8, 12, 16, 20]
    finger_mcp = [5, 9, 13, 17]
    count = 0
    for tip, mcp in zip(finger_tips, finger_mcp):
        if landmarks[tip].y < landmarks[mcp].y:  # tip higher than base
            count += 1
    return count >= 3  # At least 3 fingers extended → palm open

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Mirror
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                h, w, _ = frame.shape
                landmarks = hand_landmarks.landmark

                # Index finger tip
                x = int(landmarks[8].x * w)
                y = int(landmarks[8].y * h)

                # Erase if palm is open
                if is_palm_open(landmarks):
                    cv2.circle(canvas, (x, y), 40, (255, 255, 255), -1)  # Eraser effect
                    prev_x, prev_y = None, None  # reset line
                else:
                    # Draw line with index finger
                    if prev_x is not None and prev_y is not None:
                        cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 0, 255), 5)
                    prev_x, prev_y = x, y

                # Draw fingertip marker
                cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)

        else:
            prev_x, prev_y = None, None

        # Merge drawing with webcam
        combined = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)

        cv2.imshow("Finger Drawing + Erase", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # quit
            break
        elif key == ord('c'):  # clear all
            canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255

cap.release()
cv2.destroyAllWindows()
