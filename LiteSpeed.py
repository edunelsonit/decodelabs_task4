import os
import cv2
import numpy as np
import pytesseract
import math
import urllib.request
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Model asset paths and download URLs
MODELS = {
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
}

def ensure_models_exist():
    for name, url in MODELS.items():
        if not os.path.exists(name):
            print(f"Downloading {name}...")
            urllib.request.urlretrieve(url, name)

class TorchFreeVisionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Lightweight Multi-Task Vision (MediaPipe Tasks)")
        self.root.geometry("950x800")

        ensure_models_exist()

        # Initialize Hand Landmarker Task
        hand_base = python.BaseOptions(model_asset_path="hand_landmarker.task")
        hand_options = vision.HandLandmarkerOptions(
            base_options=hand_base,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

        # Initialize Face Landmarker Task
        face_base = python.BaseOptions(model_asset_path="face_landmarker.task")
        face_options = vision.FaceLandmarkerOptions(
            base_options=face_base,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(face_options)

        # Setup GUI Components
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side="top", fill="x", padx=10, pady=10)

        self.btn_start = ttk.Button(control_frame, text="▶️ Start Camera", command=self.start_camera)
        self.btn_start.pack(side="left", padx=5)

        self.btn_stop = ttk.Button(control_frame, text="⏹️ Stop Camera", command=self.stop_camera, state="disabled")
        self.btn_stop.pack(side="left", padx=5)

        self.video_canvas = tk.Label(self.root, text="Camera Stream Offline", bg="#1e272e", fg="white")
        self.video_canvas.pack(fill="both", expand=True, padx=10, pady=5)

        out_frame = ttk.LabelFrame(self.root, text=" Live Detection Metrics ")
        out_frame.pack(side="bottom", fill="x", padx=10, pady=10)

        self.metrics_lbl = ttk.Label(out_frame, text="Click Start to process video.", font=("Helvetica", 10, "bold"))
        self.metrics_lbl.pack(anchor="w", padx=10, pady=10)

        self.cap = None
        self.is_running = False
        self.frame_count = 0
        self.last_detected_text = "None"

    def count_fingers(self, landmarks, handedness_label):
        tip_ids = [4, 8, 12, 16, 20]
        fingers = []

        # Thumb
        if handedness_label == "Right":
            fingers.append(1 if landmarks[tip_ids[0]].x < landmarks[tip_ids[0] - 1].x else 0)
        else:
            fingers.append(1 if landmarks[tip_ids[0]].x > landmarks[tip_ids[0] - 1].x else 0)

        # 4 Fingers
        for tid in tip_ids[1:]:
            fingers.append(1 if landmarks[tid].y < landmarks[tid - 2].y else 0)

        return sum(fingers)

    def estimate_expression(self, landmarks):
        mouth_w = math.dist((landmarks[61].x, landmarks[61].y), (landmarks[291].x, landmarks[291].y))
        mouth_h = math.dist((landmarks[13].x, landmarks[13].y), (landmarks[14].x, landmarks[14].y))
        mar = mouth_h / (mouth_w + 1e-6)

        if mar > 0.35:
            return "Surprised 😲"
        elif mar > 0.18:
            return "Smiling 😀"
        else:
            return "Neutral 😐"

    def process_frame(self, frame):
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # A. Hands Detection
        hand_result = self.hand_landmarker.detect(mp_image)
        total_fingers = 0
        if hand_result and hand_result.hand_landmarks:
            for i, landmarks in enumerate(hand_result.hand_landmarks):
                handedness_label = hand_result.handedness[i][0].category_name if hand_result.handedness else "Right"
                count = self.count_fingers(landmarks, handedness_label)
                total_fingers += count
                for lm in landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

        # B. Facial Expression
        face_result = self.face_landmarker.detect(mp_image)
        expression = "None"
        if face_result and face_result.face_landmarks:
            landmarks = face_result.face_landmarks[0]
            expression = self.estimate_expression(landmarks)
            cv2.putText(frame, f"Expression: {expression}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        # C. OCR Text (Tesseract every 30 frames)
        if self.frame_count % 30 == 0:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                text = pytesseract.image_to_string(gray, config='--psm 6').strip()
                if text:
                    self.last_detected_text = text.replace('\n', ' ')[:40]
            except Exception:
                self.last_detected_text = "Tesseract Not Configured"

        metrics = f"🖐️ Fingers: {total_fingers}  |  🎭 Emotion: {expression}  |  🔤 OCR Text: {self.last_detected_text}"
        return frame, metrics

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not access webcam.")
            return

        self.is_running = True
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.update_loop()

    def update_loop(self):
        if not self.is_running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
            frame = cv2.flip(frame, 1)
            processed_frame, metrics = self.process_frame(frame)

            rgb_display = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(rgb_display)
            img_pil.thumbnail((750, 480))
            self.tk_img = ImageTk.PhotoImage(img_pil)
            self.video_canvas.config(image=self.tk_img, text="")
            self.metrics_lbl.config(text=metrics)

        self.root.after(30, self.update_loop)

    def stop_camera(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.video_canvas.config(image="", text="Camera Stream Offline")

    def __del__(self):
        if hasattr(self, 'hand_landmarker'):
            self.hand_landmarker.close()
        if hasattr(self, 'face_landmarker'):
            self.face_landmarker.close()

if __name__ == "__main__":
    root = tk.Tk()
    app = TorchFreeVisionApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop_camera(), root.destroy()))
    root.mainloop()