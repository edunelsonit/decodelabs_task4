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

# --- OFFICIAL MEDIAPIPE POSE MODEL URL ---
MODEL_PATH = "pose_landmarker_lite.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"

def ensure_model_downloaded():
    """Downloads model and verifies it is a valid binary file (> 100KB)."""
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) < 100000:
        print(f"Removing corrupted file: {MODEL_PATH}")
        os.remove(MODEL_PATH)

    if not os.path.exists(MODEL_PATH):
        print(f"Downloading model asset: {MODEL_PATH}...")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print(f"Downloaded {MODEL_PATH} ({os.path.getsize(MODEL_PATH)} bytes).")
        except Exception as e:
            print(f"Error downloading {MODEL_PATH}: {e}")
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)


class UserActionRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time User Action & Activity Recognition App")
        self.root.geometry("900x780")

        # 1. Ensure Model Asset
        ensure_model_downloaded()

        # 2. Initialize Pose Landmarker Engine
        pose_base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        pose_options = vision.PoseLandmarkerOptions(
            base_options=pose_base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=2,  # Supports tracking up to 2 users simultaneously
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5
        )
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

        # 3. GUI Layout
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side="top", fill="x", padx=10, pady=10)

        self.btn_start = ttk.Button(control_frame, text="▶️ Start Live Camera", command=self.start_camera)
        self.btn_start.pack(side="left", padx=5)

        self.btn_stop = ttk.Button(control_frame, text="⏹️ Stop Camera", command=self.stop_camera, state="disabled")
        self.btn_stop.pack(side="left", padx=5)

        self.status_lbl = ttk.Label(control_frame, text="Status: Offline", font=("Helvetica", 10))
        self.status_lbl.pack(side="left", padx=15)

        self.video_canvas = tk.Label(self.root, text="Camera Feed Offline", bg="#1e272e", fg="white")
        self.video_canvas.pack(fill="both", expand=True, padx=10, pady=5)

        # Analytics Dashboard
        out_frame = ttk.LabelFrame(self.root, text=" Real-Time Action Analytics ")
        out_frame.pack(side="bottom", fill="x", padx=10, pady=10)

        self.metrics_lbl = ttk.Label(
            out_frame, 
            text="Start camera feed to observe user action detection.", 
            font=("Helvetica", 11, "bold")
        )
        self.metrics_lbl.pack(anchor="w", padx=10, pady=10)

        self.cap = None
        self.is_running = False

    def calculate_angle(self, a, b, c):
        """Calculates 2D angle (in degrees) between three joint keypoints: a(x,y), b(x,y), c(x,y)."""
        ang = math.degrees(
            math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x)
        )
        return abs(ang if ang >= 0 else ang + 360)

    def classify_action(self, lm):
        """
        Rule-based Action Recognition Engine:
        Maps 33 3D Pose landmarks to classify human activities.
        Indices:
          Nose: 0 | Shoulders: 11, 12 | Wrists: 15, 16 
          Hips: 23, 24 | Knees: 25, 26 | Ankles: 27, 28
        """
        # Key landmark shortcuts
        nose = lm[0]
        left_shoulder, right_shoulder = lm[11], lm[12]
        left_wrist, right_wrist = lm[15], lm[16]
        left_hip, right_hip = lm[23], lm[24]
        left_knee, right_knee = lm[25], lm[26]
        left_ankle, right_ankle = lm[27], lm[28]

        # Calculate Joint Angles
        left_knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = self.calculate_angle(right_hip, right_knee, right_ankle)
        avg_knee_angle = (left_knee_angle + right_knee_angle) / 2.0

        avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2.0
        avg_hip_y = (left_hip.y + right_hip.y) / 2.0

        # Action 1: Raising Both Hands
        if left_wrist.y < left_shoulder.y and right_wrist.y < right_shoulder.y:
            return "Raising Both Hands 🙋‍♂️"

        # Action 2: Waving One Hand
        if (left_wrist.y < left_shoulder.y and right_wrist.y > right_shoulder.y) or \
           (right_wrist.y < right_shoulder.y and left_wrist.y > left_shoulder.y):
            return "Waving / One Hand Raised 👋"

        # Action 3: Lying Down / Horizontal Orientation
        body_angle = abs(left_shoulder.y - left_hip.y)
        if body_angle < 0.12 and abs(left_shoulder.x - left_hip.x) > 0.2:
            return "Lying Down 🛌"

        # Action 4: Sitting vs Standing (Knee angle bending threshold)
        if 70 <= avg_knee_angle <= 130 or (avg_hip_y - avg_shoulder_y < 0.15):
            return "Sitting 🪑"

        # Action 5: Bowing / Leaning Forward
        if nose.y > avg_hip_y:
            return "Bowing / Leaning Down 🙇"

        # Default Action
        return "Standing 🚶"

    def process_frame(self, frame):
        """Processes frame through PoseLandmarker and renders annotations."""
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        pose_result = self.pose_landmarker.detect(mp_image)
        actions = []

        POSE_CONNECTIONS = [
            (11,12), (11,13), (13,15), (12,14), (14,16),  # Arms
            (11,23), (12,24), (23,24),                   # Torso
            (23,25), (24,26), (25,27), (26,28)           # Legs
        ]

        if pose_result and pose_result.pose_landmarks:
            for idx, landmarks in enumerate(pose_result.pose_landmarks):
                # Detect action for current user
                action = self.classify_action(landmarks)
                actions.append(f"User {idx+1}: {action}")

                # Draw Pose Skeleton
                for start_idx, end_idx in POSE_CONNECTIONS:
                    p1 = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
                    p2 = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
                    cv2.line(frame, p1, p2, (0, 255, 0), 2)

                # Draw Joint Keypoints
                for lm in landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

                # Overlay Action Label above head
                head_x = int(landmarks[0].x * w)
                head_y = int(landmarks[0].y * h)
                cv2.putText(
                    frame, 
                    f"User {idx+1}: {action}", 
                    (max(10, head_x - 60), max(30, head_y - 30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 255, 255), 
                    2
                )

        return frame, actions

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
            processed_frame, actions = self.process_frame(frame)

            # Render frame to GUI
            rgb_display = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(rgb_display)
            img_pil.thumbnail((750, 480))
            self.tk_img = ImageTk.PhotoImage(img_pil)
            self.video_canvas.config(image=self.tk_img, text="")

            # Update Metrics Panel
            if actions:
                summary_text = "  |  ".join(actions)
            else:
                summary_text = "No users detected in frame."

            self.metrics_lbl.config(text=f"📋 Detected Actions: {summary_text}")

        self.root.after(30, self.update_loop)

    def stop_camera(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None

        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.status_lbl.config(text="Status: Offline")
        self.video_canvas.config(image="", text="Camera Feed Offline")

    def __del__(self):
        if hasattr(self, 'pose_landmarker'):
            self.pose_landmarker.close()


if __name__ == "__main__":
    root = tk.Tk()
    app = UserActionRecognitionApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop_camera(), root.destroy()))
    root.mainloop()