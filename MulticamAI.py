import os
import math
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def get_available_cameras(max_tested=5):
    """Scans system for available video capture device indices."""
    available_cams = []
    for i in range(max_tested):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Test reading a frame to verify camera accessibility
            ret, _ = cap.read()
            if ret:
                available_cams.append(f"Camera {i}")
            cap.release()
    return available_cams if available_cams else ["Camera 0"]


class MultiCameraActionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Camera Action & Pose Detection GUI")
        self.root.geometry("900x780")

        # Initialize MediaPipe Pose Landmarker
        pose_base_options = python.BaseOptions(model_asset_path="pose_landmarker_lite.task")
        pose_options = vision.PoseLandmarkerOptions(
            base_options=pose_base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=2,
            min_pose_detection_confidence=0.5
        )
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

        # Control Panel Frame
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side="top", fill="x", padx=10, pady=10)

        # 1. CAMERA SELECTION DROPDOWN
        ttk.Label(control_frame, text="Select Camera:", font=("Helvetica", 10, "bold")).pack(side="left", padx=5)
        
        self.camera_list = get_available_cameras()
        self.selected_camera = tk.StringVar(value=self.camera_list[0])
        
        self.cam_dropdown = ttk.Combobox(
            control_frame, 
            textvariable=self.selected_camera, 
            values=self.camera_list, 
            state="readonly", 
            width=12
        )
        self.cam_dropdown.pack(side="left", padx=5)

        # 2. CONTROLS
        self.btn_start = ttk.Button(control_frame, text="▶️ Start Camera", command=self.start_camera)
        self.btn_start.pack(side="left", padx=5)

        self.btn_stop = ttk.Button(control_frame, text="⏹️ Stop Camera", command=self.stop_camera, state="disabled")
        self.btn_stop.pack(side="left", padx=5)

        self.status_lbl = ttk.Label(control_frame, text="Status: Offline", font=("Helvetica", 10))
        self.status_lbl.pack(side="left", padx=15)

        # Display Canvas
        self.video_canvas = tk.Label(self.root, text="Camera Feed Offline", bg="#1e272e", fg="white")
        self.video_canvas.pack(fill="both", expand=True, padx=10, pady=5)

        # Metrics Footer
        out_frame = ttk.LabelFrame(self.root, text=" Action Metrics ")
        out_frame.pack(side="bottom", fill="x", padx=10, pady=10)

        self.metrics_lbl = ttk.Label(out_frame, text="Select a camera and click Start.", font=("Helvetica", 11, "bold"))
        self.metrics_lbl.pack(anchor="w", padx=10, pady=10)

        self.cap = None
        self.is_running = False

    def start_camera(self):
        # Extract index number from selection (e.g., "Camera 1" -> 1)
        selected_text = self.selected_camera.get()
        cam_index = int(selected_text.split()[-1])

        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", f"Could not access {selected_text}!")
            return

        self.is_running = True
        self.btn_start.config(state="disabled")
        self.cam_dropdown.config(state="disabled")  # Lock dropdown during live stream
        self.btn_stop.config(state="normal")
        self.status_lbl.config(text=f"Status: Streaming {selected_text}")
        self.update_loop()

    def update_loop(self):
        if not self.is_running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if ret:
            # Process image and update canvas
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            pose_result = self.pose_landmarker.detect(mp_image)

            # Draw skeletons if pose detected
            if pose_result and pose_result.pose_landmarks:
                h, w, _ = frame.shape
                for landmarks in pose_result.pose_landmarks:
                    for lm in landmarks:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

            # Render frame to GUI
            rgb_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(rgb_display)
            img_pil.thumbnail((750, 480))
            self.tk_img = ImageTk.PhotoImage(img_pil)
            self.video_canvas.config(image=self.tk_img, text="")

        self.root.after(30, self.update_loop)

    def stop_camera(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None

        self.btn_start.config(state="normal")
        self.cam_dropdown.config(state="readonly")  # Unlock dropdown
        self.btn_stop.config(state="disabled")
        self.status_lbl.config(text="Status: Offline")
        self.video_canvas.config(image="", text="Camera Feed Offline")

    def __del__(self):
        if hasattr(self, 'pose_landmarker'):
            self.pose_landmarker.close()

if __name__ == "__main__":
    root = tk.Tk()
    app = MultiCameraActionApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop_camera(), root.destroy()))
    root.mainloop()