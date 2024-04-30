import mediapipe as mp
import cv2
import pyautogui

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

prev_pinch = False
prev_dist = -1

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        continue

    # Convert the BGR image to RGB and process it with MediaPipe Hands
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame)

    # Draw the hand annotations on the image.
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            distance = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5
            print(distance)

            PINCH_THRESHOLD_MIN = 0.03
            PINCH_THRESHOLD_MAX = 0.06

            # Distance threshold
            is_pinching = PINCH_THRESHOLD_MIN <= distance <= PINCH_THRESHOLD_MAX

            # If the previous state was not a pinch, and now it is, perform left click
            if not prev_pinch and is_pinching and distance < prev_dist:
                pyautogui.click(button="left")
                thumb_tip_x, thumb_tip_y = int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0])
                index_tip_x, index_tip_y = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])
                cv2.line(frame, (thumb_tip_x, thumb_tip_y), (index_tip_x, index_tip_y), (0, 255, 0), 2)

            prev_pinch = is_pinching
            prev_dist = distance

    # display
    cv2.imshow("MediaPipe Hands", frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
