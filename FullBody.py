import os
import math
import urllib.request
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- OFFICIAL MEDIAPIPE MODEL ASSET URLS ---
MODELS = {
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
}

def ensure_models_downloaded():
    """Downloads models and verifies they are valid binary files (> 100KB)."""
    for filename, url in MODELS.items():
        if os.path.exists(filename) and os.path.getsize(filename) < 100000:
            print(f"Removing corrupted file: {filename}")
            os.remove(filename)

        if not os.path.exists(filename):
            print(f"Downloading model asset: {filename}...")
            try:
                urllib.request.urlretrieve(url, filename)
                print(f"Downloaded {filename} ({os.path.getsize(filename)} bytes).")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
                if os.path.exists(filename):
                    os.remove(filename)


class FacialExpressionAndPoseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Modern MediaPipe - Facial Expression & Full Body Pose App")
        self.root.geometry("900x780")

        # 1. Download & Verify Model Assets
        ensure_models_downloaded()

        # 2. Initialize Pose Landmarker (Full Body)
        pose_base_options = python.BaseOptions(model_asset_path="pose_landmarker_lite.task")
        pose_options = vision.PoseLandmarkerOptions(
            base_options=pose_base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5
        )
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

        # 3. Initialize Face Landmarker (Facial Mesh & Expressions)
        face_base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
        face_options = vision.FaceLandmarkerOptions(
            base_options=face_base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(face_options)

        # 4. GUI Layout
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side="top", fill="x", padx=10, pady=10)

        self.btn_start = ttk.Button(control_frame, text="▶️ Start Live Camera", command=self.start_camera)
        self.btn_start.pack(side="left", padx=5)

        self.btn_stop = ttk.Button(control_frame, text="⏹️ Stop Camera", command=self.stop_camera, state="disabled")
        self.btn_stop.pack(side="left", padx=5)

        self.status_lbl = ttk.Label(control_frame, text="Status: Offline", font=("Helvetica", 10))
        self.status_lbl.pack(side="left", padx=15)

        self.video_canvas = tk.Label(self.root, text="Camera Stream Offline", bg="#1e272e", fg="white")
        self.video_canvas.pack(fill="both", expand=True, padx=10, pady=5)

        # Output Analytics Dashboard
        out_frame = ttk.LabelFrame(self.root, text=" Real-Time Expression & Pose Metrics ")
        out_frame.pack(side="bottom", fill="x", padx=10, pady=10)

        self.metrics_lbl = ttk.Label(out_frame, text="Start camera feed to observe real-time expression & pose tracking.", font=("Helvetica", 11, "bold"))
        self.metrics_lbl.pack(anchor="w", padx=10, pady=10)

        self.cap = None
        self.is_running = False

    def detect_expression(self, landmarks):
        """
        Estimates facial expression based on geometric ratios from key 3D facial mesh points:
        - Outer lip corners: 61, 291
        - Upper / lower lip center: 13, 14
        - Eye landmarks: Upper/Lower lids (159/145) and Eye corners (33/133)
        """
        # Mouth Width (Corner to Corner)
        mouth_width = math.dist(
            (landmarks[61].x, landmarks[61].y),
            (landmarks[291].x, landmarks[291].y)
        )
        # Mouth Height (Upper Lip to Lower Lip)
        mouth_height = math.dist(
            (landmarks[13].x, landmarks[13].y),
            (landmarks[14].x, landmarks[14].y)
        )
        
        # Mouth Aspect Ratio (MAR)
        mar = mouth_height / (mouth_width + 1e-6)

        # Eye Opening Height (Left Eye)
        left_eye_height = math.dist(
            (landmarks[159].x, landmarks[159].y),
            (landmarks[145].x, landmarks[145].y)
        )
        left_eye_width = math.dist(
            (landmarks[33].x, landmarks[33].y),
            (landmarks[133].x, landmarks[133].y)
        )
        ear = left_eye_height / (left_eye_width + 1e-6)

        # Rule-based Expression Classifier
        if mar > 0.4 and ear > 0.35:
            return "Surprised 😲", mar
        elif mar > 0.18:
            return "Smiling 😀", mar
        else:
            return "Neutral 😐", mar

    def process_frame(self, frame):
        """Processes live camera frame through Pose and Face Landmarkers."""
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        detected_expression = "None Detected"
        pose_detected = False

        # --- A. FACIAL EXPRESSION RECOGNITION ---
        face_result = self.face_landmarker.detect(mp_image)
        if face_result and face_result.face_landmarks:
            landmarks = face_result.face_landmarks[0]
            detected_expression, mar_score = self.detect_expression(landmarks)

            # Draw key facial outline contours (lips & eyes)
            key_points = [61, 291, 13, 14, 159, 145, 33, 133]
            for idx in key_points:
                cx, cy = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
                cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)

            # Overlay expression label above face (using nose tip landmark 1)
            nose_x, nose_y = int(landmarks[1].x * w), int(landmarks[1].y * h)
            cv2.putText(frame, f"Expression: {detected_expression}", 
                        (max(10, nose_x - 80), max(30, nose_y - 120)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # --- B. FULL BODY POSE DETECTION ---
        pose_result = self.pose_landmarker.detect(mp_image)
        
        # MediaPipe Pose Connections (33 Landmarks)
        POSE_CONNECTIONS = [
            (0,1), (1,2), (2,3), (3,7), (0,4), (4,5), (5,6), (6,8),
            (9,10), (11,12), (11,13), (13,15), (12,14), (14,16),
            (11,23), (12,24), (23,24), (23,25), (24,26), (25,27), (26,28)
        ]

        if pose_result and pose_result.pose_landmarks:
            pose_detected = True
            landmarks = pose_result.pose_landmarks[0]

            # Draw Pose Connections
            for start_idx, end_idx in POSE_CONNECTIONS:
                p1 = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
                p2 = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
                cv2.line(frame, p1, p2, (255, 105, 180), 2)  # Hot pink skeleton lines

            # Draw Pose Keypoints
            for lm in landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 165, 255), -1)

        return frame, detected_expression, pose_detected

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Could not access webcam device.")
            return

        self.is_running = True
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.status_lbl.config(text="Status: Streaming Live Camera Feed")
        self.update_loop()

    def update_loop(self):
        if not self.is_running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if ret:
            processed_frame, expression, pose_found = self.process_frame(frame)

            # Render frame to Tkinter Canvas
            rgb_display = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(rgb_display)
            img_pil.thumbnail((750, 480))
            self.tk_img = ImageTk.PhotoImage(img_pil)
            self.video_canvas.config(image=self.tk_img, text="")

            # Update Metrics Panel
            pose_status = "Detected ✅" if pose_found else "Not Detected ❌"
            self.metrics_lbl.config(
                text=f"🎭 Expression: {expression}  |  🧍 Full Body Pose: {pose_status}"
            )

        self.root.after(30, self.update_loop)

    def stop_camera(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None

        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.status_lbl.config(text="Status: Offline")
        self.video_canvas.config(image="", text="Camera Stream Offline")

    def __del__(self):
        if hasattr(self, 'pose_landmarker'):
            self.pose_landmarker.close()
        if hasattr(self, 'face_landmarker'):
            self.face_landmarker.close()


if __name__ == "__main__":
    root = tk.Tk()
    app = FacialExpressionAndPoseApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop_camera(), root.destroy()))
    root.mainloop()