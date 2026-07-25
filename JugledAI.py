import cv2
import numpy as np
import mediapipe as mp
import easyocr
from deepface import DeepFace

# ==========================================
# 1. Initialization
# ==========================================

# Initialize EasyOCR Reader (loads GPU if available, else CPU)
print("Loading OCR Engine...")
ocr_reader = easyocr.Reader(['en'], gpu=False)

# Initialize MediaPipe Solutions
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
mp_object_detection = mp.solutions.object_detection
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

face_detector = mp_face_detection.FaceDetection(
    min_detection_confidence=0.6
)

object_detector = mp_object_detection.ObjectDetection(
    min_detection_confidence=0.5,
    model_selection=0 # 0 for short-range objects (within 2m)
)

# Finger Tip IDs for counting (Thumb, Index, Middle, Ring, Pinky)
TIP_IDS = [4, 8, 12, 16, 20]

def count_fingers(hand_landmarks, hand_label):
    """Counts extended fingers given hand landmarks and hand side (Left/Right)."""
    lm = hand_landmarks.landmark
    fingers = []

    # 1. Thumb logic (horizontal check based on hand side)
    if hand_label == "Right":
        fingers.append(1 if lm[TIP_IDS[0]].x < lm[TIP_IDS[0] - 1].x else 0)
    else:
        fingers.append(1 if lm[TIP_IDS[0]].x > lm[TIP_IDS[0] - 1].x else 0)

    # 2. Four Fingers logic (vertical check: tip higher than PIP joint)
    for tip_id in TIP_IDS[1:]:
        fingers.append(1 if lm[tip_id].y < lm[tip_id - 2].y else 0)

    return sum(fingers)


# ==========================================
# 2. Main Video Stream Loop
# ==========================================

cap = cv2.VideoCapture(0)
frame_count = 0

print("\nStarting All-in-One Vision Stream. Press 'q' to exit.\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    # Mirror frame for natural interaction
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    
    # Convert BGR to RGB for MediaPipe / DeepFace
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --------------------------------------
    # A. Hands & Finger Counting
    # --------------------------------------
    hand_results = hands.process(rgb_frame)
    total_fingers = 0

    if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
        for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            label = handedness.classification[0].label # "Left" or "Right"
            finger_count = count_fingers(hand_landmarks, label)
            total_fingers += finger_count

            # Wrist position for label placement
            wrist_x = int(hand_landmarks.landmark[0].x * w)
            wrist_y = int(hand_landmarks.landmark[0].y * h)
            cv2.putText(frame, f"{label}: {finger_count}", (wrist_x - 30, wrist_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(frame, f"Total Fingers: {total_fingers}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # --------------------------------------
    # B. Object Detection
    # --------------------------------------
    obj_results = object_detector.process(rgb_frame)
    if obj_results.detections:
        for detection in obj_results.detections:
            bbox = detection.location_data.relative_bounding_box
            xmin = int(bbox.xmin * w)
            ymin = int(bbox.ymin * h)
            box_w = int(bbox.width * w)
            box_h = int(bbox.height * h)

            label_name = detection.label[0] if detection.label else "Object"
            score = int(detection.score[0] * 100)

            cv2.rectangle(frame, (xmin, ymin), (xmin + box_w, ymin + box_h), (255, 165, 0), 2)
            cv2.putText(frame, f"{label_name} {score}%", (xmin, max(20, ymin - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)

    # --------------------------------------
    # C. Face Detection & Expression Recognition
    # --------------------------------------
    # Run heavy emotion recognition every 10th frame to preserve FPS
    if frame_count % 10 == 0:
        try:
            face_analysis = DeepFace.analyze(
                img_path=rgb_frame,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )
            if face_analysis:
                primary_face = face_analysis[0]
                dominant_emotion = primary_face['dominant_emotion']
                region = primary_face['region']
                
                fx, fy, fw, fh = region['x'], region['y'], region['w'], region['h']
                cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 0, 255), 2)
                cv2.putText(frame, f"Emotion: {dominant_emotion}", (fx, max(20, fy - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        except Exception:
            pass

    # --------------------------------------
    # D. Text Recognition (OCR)
    # --------------------------------------
    # Run OCR every 30th frame (approx. once per second)
    if frame_count % 30 == 0:
        ocr_results = ocr_reader.readtext(frame)
        for (bbox_pts, text, prob) in ocr_results:
            if prob > 0.4:
                pt1 = tuple(map(int, bbox_pts[0]))
                pt2 = tuple(map(int, bbox_pts[2]))
                cv2.rectangle(frame, pt1, pt2, (0, 255, 255), 2)
                cv2.putText(frame, f"Text: {text}", (pt1[0], max(20, pt1[1] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Output window display
    cv2.imshow("Multi-Task Computer Vision Pipeline", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()