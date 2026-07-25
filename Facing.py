import os
import time
import urllib.request
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- 1. DOWNLOAD TASK MODEL IF NOT PRESENT ---
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

def ensure_model_downloaded():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand_landmarker.task model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")


class ModernMediaPipeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Modern MediaPipe Tasks API - Hand & Finger Counting")
        self.root.geometry("850x720")

        # Ensure model asset exists
        ensure_model_downloaded()

        # Shared state across threads for async callback
        self.latest_result = None
        self.latest_frame = None

        # --- 2. INITIALIZE MEDIAPIPE HAND LANDMARKER (TASKS API) ---
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
            result_callback=self.result_callback
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

        # --- 3. GUI LAYOUT ---
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side="top", fill="x", padx=10, pady=10)

        self.btn_start = ttk.Button(control_frame, text="▶️ Start Camera", command=self.start_camera)
        self.btn_start.pack(side="left", padx=5)

        self.btn_stop = ttk.Button(control_frame, text="⏹️ Stop Camera", command=self.stop_camera, state="disabled")
        self.btn_stop.pack(side="left", padx=5)

        self.status_lbl = ttk.Label(control_frame, text="Status: Offline", font=("Helvetica", 10))
        self.status_lbl.pack(side="left", padx=15)

        self.video_canvas = tk.Label(self.root, text="Camera Stream Stopped", bg="#1e272e", fg="white")
        self.video_canvas.pack(fill="both", expand=True, padx=10, pady=5)

        out_frame = ttk.LabelFrame(self.root, text=" Recognition Metrics ")
        out_frame.pack(side="bottom", fill="x", padx=10, pady=10)

        self.metrics_lbl = ttk.Label(out_frame, text="Start the camera to view hand detection data.", font=("Helvetica", 11, "bold"))
        self.metrics_lbl.pack(anchor="w", padx=10, pady=10)

        self.cap = None
        self.is_running = False

    def result_callback(self, result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        """Asynchronous result listener called by MediaPipe Tasks engine."""
        self.latest_result = result

    def count_raised_fingers(self, hand_landmarks, handedness):
        """Counts raised fingers based on landmark coordinates."""
        tip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        fingers = []

        # Thumb logic based on Handedness category
        if handedness == "Right":
            fingers.append(1 if hand_landmarks[tip_ids[0]].x < hand_landmarks[tip_ids[0] - 1].x else 0)
        else:
            fingers.append(1 if hand_landmarks[tip_ids[0]].x > hand_landmarks[tip_ids[0] - 1].x else 0)

        # 4 Fingers logic (vertical Y height)
        for tid in range(1, 5):
            if hand_landmarks[tip_ids[tid]].y < hand_landmarks[tip_ids[tid] - 2].y:
                fingers.append(1)
            else:
                fingers.append(0)

        return sum(fingers)

    def draw_landmarks_and_measure(self, frame):
        """Annotates frame with hand connections and counts raised fingers."""
        if self.latest_result is None or not self.latest_result.hand_landmarks:
            return frame, 0, 0

        h, w, _ = frame.shape
        total_hands = len(self.latest_result.hand_landmarks)
        total_fingers = 0

        # Define 21 hand skeleton connections
        HAND_CONNECTIONS = [
            (0,1), (1,2), (2,3), (3,4),
            (0,5), (5,6), (6,7), (7,8),
            (5,9), (9,10), (10,11), (11,12),
            (9,13), (13,14), (14,15), (15,16),
            (13,17), (0,17), (17,18), (18,19), (19,20)
        ]

        for idx, landmarks in enumerate(self.latest_result.hand_landmarks):
            # Extract Handedness label (Left/Right)
            handedness = "Right"
            if self.latest_result.handedness and idx < len(self.latest_result.handedness):
                handedness = self.latest_result.handedness[idx][0].category_name

            # Count fingers
            finger_count = self.count_raised_fingers(landmarks, handedness)
            total_fingers += finger_count

            # Draw Connections & Keypoints
            for start_idx, end_idx in HAND_CONNECTIONS:
                p1 = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
                p2 = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
                cv2.line(frame, p1, p2, (0, 255, 0), 2)

            for lm in landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

            # Text annotation near wrist point
            wrist_x, wrist_y = int(landmarks[0].x * w), int(landmarks[0].y * h)
            cv2.putText(frame, f"{handedness}: {finger_count} Fingers", 
                        (wrist_x - 30, wrist_y + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return frame, total_hands, total_fingers

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Could not access webcam device.")
            return

        self.is_running = True
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.status_lbl.config(text="Status: Live Camera Streaming")
        self.update_loop()

    def update_loop(self):
        if not self.is_running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if ret:
            # Send frame to MediaPipe Tasks asynchronously
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            frame_timestamp_ms = int(time.time() * 1000)
            
            self.landmarker.detect_async(mp_image, frame_timestamp_ms)

            # Annotate current frame with latest async results
            annotated_frame, hands_count, finger_count = self.draw_landmarks_and_measure(frame)

            # Render to Tkinter
            rgb_display = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(rgb_display)
            img_pil.thumbnail((750, 480))
            self.tk_img = ImageTk.PhotoImage(img_pil)
            self.video_canvas.config(image=self.tk_img, text="")

            # Update Metrics Panel
            self.metrics_lbl.config(
                text=f"✋ Hands Detected: {hands_count}  |  🖐️ Total Raised Fingers: {finger_count}"
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
        self.video_canvas.config(image="", text="Camera Stream Stopped")

    def __del__(self):
        if hasattr(self, 'landmarker'):
            self.landmarker.close()


if __name__ == "__main__":
    root = tk.Tk()
    app = ModernMediaPipeApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop_camera(), root.destroy()))
    root.mainloop()