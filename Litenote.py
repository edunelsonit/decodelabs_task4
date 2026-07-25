import cv2
import numpy as np
import pytesseract
import math
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Optional: Set Tesseract path explicitly on Windows if not added to System PATH
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class TorchFreeVisionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Lightweight Multi-Task Vision (No PyTorch)")
        self.root.geometry("950x800")

        # ----------------------------------------------
        # Initialize MediaPipe Solutions (C++ / TFLite backend)
        # ----------------------------------------------
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

        self.mp_object_detector = mp.solutions.object_detection
        self.object_detector = self.mp_object_detector.ObjectDetection(
            min_detection_confidence=0.5
        )

        self.mp_drawing = mp.solutions.drawing_utils

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

    def count_fingers(self, landmarks, hand_label):
        """Calculates extended fingers using joint vectors."""
        tip_ids = [4, 8, 12, 16, 20]
        fingers = []

        # Thumb
        if hand_label == "Right":
            fingers.append(1 if landmarks[tip_ids[0]].x < landmarks[tip_ids[0] - 1].x else 0)
        else:
            fingers.append(1 if landmarks[tip_ids[0]].x > landmarks[tip_ids[0] - 1].x else 0)

        # 4 Fingers
        for tid in tip_ids[1:]:
            fingers.append(1 if landmarks[tid].y < landmarks[tid - 2].y else 0)

        return sum(fingers)

    def estimate_expression(self, landmarks):
        """Geometry-based lightweight emotion classification."""
        # Lip aspect ratio
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
        
        # A. Hands & Finger Count
        hand_results = self.hands.process(rgb_frame)
        total_fingers = 0
        if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
            for landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                label = handedness.classification[0].label
                count = self.count_fingers(landmarks.landmark, label)
                total_fingers += count
                self.mp_drawing.draw_landmarks(frame, landmarks, self.mp_hands.HAND_CONNECTIONS)

        # B. Facial Expression Recognition (Geometry)
        face_results = self.face_mesh.process(rgb_frame)
        expression = "None"
        if face_results.multi_face_landmarks:
            landmarks = face_results.multi_face_landmarks[0].landmark
            expression = self.estimate_expression(landmarks)
            cv2.putText(frame, f"Expression: {expression}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        # C. Object Detection
        obj_results = self.object_detector.process(rgb_frame)
        if obj_results.detections:
            for detection in obj_results.detections:
                bbox = detection.location_data.relative_bounding_box
                xmin, ymin = int(bbox.xmin * w), int(bbox.ymin * h)
                box_w, box_h = int(bbox.width * w), int(bbox.height * h)
                cv2.rectangle(frame, (xmin, ymin), (xmin + box_w, ymin + box_h), (255, 165, 0), 2)

        # D. Text Recognition (OCR via Tesseract every 30 frames)
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

if __name__ == "__main__":
    root = tk.Tk()
    app = TorchFreeVisionApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop_camera(), root.destroy()))
    root.mainloop()