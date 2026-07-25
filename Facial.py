import os
import urllib.request
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- MODEL DOWNLOAD CONFIGURATION ---
MODELS = {
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "face_detector.tflite": "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
}

def ensure_models_downloaded():
    """Downloads required MediaPipe Tasks model files if not already present."""
    for filename, url in MODELS.items():
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, filename)
                print(f"Downloaded {filename} successfully.")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")


class ModernFaceAndHandApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Modern MediaPipe - Face Detection & Finger Counting GUI")
        self.root.geometry("900x750")

        # 1. Download Model Assets
        ensure_models_downloaded()

        # 2. Initialize Modern MediaPipe Hand Landmarker
        hand_base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
        hand_options = vision.HandLandmarkerOptions(
            base_options=hand_base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

        # 3. Initialize Modern MediaPipe Face Detector
        face_base_options = python.BaseOptions(model_asset_path="face_detector.tflite")
        face_options = vision.FaceDetectorOptions(
            base_options=face_base_options,
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=0.5
        )
        self.face_detector = vision.FaceDetector.create_from_options(face_options)

        # 4. GUI Layout
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side="top", fill="x", padx=10, pady=10)

        self.btn_start = ttk.Button(control_frame, text="▶️ Start Live Camera", command=self.start_camera)
        self.btn_start.pack(side="left", padx=5)

        self.btn_stop = ttk.Button(control_frame, text="⏹️ Stop Camera", command=self.stop_camera, state="disabled")
        self.btn_stop.pack(side="left", padx=5)

        self.status_lbl = ttk.Label(control_frame, text="Status: Ready", font=("Helvetica", 10))
        self.status_lbl.pack(side="left", padx=15)

        # Video Frame Canvas
        self.video_canvas = tk.Label(self.root, text="Camera Feed Offline", bg="#1e272e", fg="white")
        self.video_canvas.pack(fill="both", expand=True, padx=10, pady=5)

        # Live Metrics Output Frame
        out_frame = ttk.LabelFrame(self.root, text=" Live Detection & Recognition Analytics ")
        out_frame.pack(side="bottom", fill="x", padx=10, pady=10)

        self.metrics_lbl = ttk.Label(out_frame, text="Start camera feed to observe real-time face & hand recognition.", font=("Helvetica", 11, "bold"))
        self.metrics_lbl.pack(anchor="w", padx=10, pady=10)

        self.cap = None
        self.is_running = False

    def count_raised_fingers(self, hand_landmarks, handedness):
        """Determines open/folded fingers using landmark spatial relationships."""
        tip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        fingers = []

        # Thumb logic (horizontal X displacement based on hand label)
        if handedness == "Right":
            fingers.append(1 if hand_landmarks[tip_ids[0]].x < hand_landmarks[tip_ids[0] - 1].x else 0)
        else:
            fingers.append(1 if hand_landmarks[tip_ids[0]].x > hand_landmarks[tip_ids[0] - 1].x else 0)

        # Finger logic (vertical Y coordinate height comparison)
        for tid in range(1, 5):
            if hand_landmarks[tip_ids[tid]].y < hand_landmarks[tip_ids[tid] - 2].y:
                fingers.append(1)
            else:
                fingers.append(0)

        return sum(fingers)

    def process_frame(self, frame):
        """Processes frame through Face Detector and Hand Landmarker."""
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # --- A. FACE DETECTION ---
        face_result = self.face_detector.detect(mp_image)
        face_count = 0

        if face_result and face_result.detections:
            face_count = len(face_result.detections)
            for detection in face_result.detections:
                bbox = detection.bounding_box
                x, y, bw, bh = int(bbox.origin_x), int(bbox.origin_y), int(bbox.width), int(bbox.height)
                
                # Draw bounding box and label around detected face
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (255, 0, 0), 2)
                confidence = detection.categories[0].score if detection.categories else 0.0
                cv2.putText(frame, f"Face ({confidence*100:.0f}%)", (x, max(y - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # --- B. HAND LANDMARKING & FINGER COUNTING ---
        hand_result = self.hand_landmarker.detect(mp_image)
        hands_count = 0
        total_fingers = 0

        HAND_CONNECTIONS = [
            (0,1), (1,2), (2,3), (3,4),
            (0,5), (5,6), (6,7), (7,8),
            (5,9), (9,10), (10,11), (11,12),
            (9,13), (13,14), (14,15), (15,16),
            (13,17), (0,17), (17,18), (18,19), (19,20)
        ]

        if hand_result and hand_result.hand_landmarks:
            hands_count = len(hand_result.hand_landmarks)

            for idx, landmarks in enumerate(hand_result.hand_landmarks):
                # Retrieve Left or Right Hand classification
                handedness = "Right"
                if hand_result.handedness and idx < len(hand_result.handedness):
                    handedness = hand_result.handedness[idx][0].category_name

                # Count fingers
                fingers = self.count_raised_fingers(landmarks, handedness)
                total_fingers += fingers

                # Draw skeleton links
                for start_idx, end_idx in HAND_CONNECTIONS:
                    p1 = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
                    p2 = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
                    cv2.line(frame, p1, p2, (0, 255, 0), 2)

                # Draw landmarks points
                for lm in landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

                # Label on wrist point
                wrist_x, wrist_y = int(landmarks[0].x * w), int(landmarks[0].y * h)
                cv2.putText(frame, f"{handedness}: {fingers} Fingers",
                            (wrist_x - 30, min(wrist_y + 30, h - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return frame, face_count, hands_count, total_fingers

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Could not access webcam device.")
            return

        self.is_running = True
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.status_lbl.config(text="Status: Live Camera Feed Active")
        self.update_loop()

    def update_loop(self):
        if not self.is_running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if ret:
            # Run Face + Hand recognition pipeline
            processed_frame, face_count, hands_count, finger_count = self.process_frame(frame)

            # Convert BGR -> RGB for Tkinter GUI display
            rgb_display = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(rgb_display)
            img_pil.thumbnail((750, 480))
            self.tk_img = ImageTk.PhotoImage(img_pil)
            self.video_canvas.config(image=self.tk_img, text="")

            # Update Metrics Panel
            self.metrics_lbl.config(
                text=f"😀 Faces Detected: {face_count}  |  ✋ Hands Detected: {hands_count}  |  🖐️ Total Raised Fingers: {finger_count}"
            )

        self.root.after(30, self.update_loop)

    def stop_camera(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None

        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.status_lbl.config(text="Status: Stopped")
        self.video_canvas.config(image="", text="Camera Feed Offline")

    def __del__(self):
        if hasattr(self, 'hand_landmarker'):
            self.hand_landmarker.close()
        if hasattr(self, 'face_detector'):
            self.face_detector.close()


if __name__ == "__main__":
    root = tk.Tk()
    app = ModernFaceAndHandApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop_camera(), root.destroy()))
    root.mainloop()