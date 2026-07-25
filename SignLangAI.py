import os
import urllib.request
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# MediaPipe Gesture Recognizer Model Asset
MODEL_NAME = "gesture_recognizer.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"

def ensure_gesture_model():
    """Downloads gesture recognizer model if missing."""
    if not os.path.exists(MODEL_NAME):
        print("Downloading Gesture Recognizer task asset...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_NAME)

class SignLanguageVisionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language & Gesture Recognition Suite")
        self.root.geometry("950x800")

        ensure_gesture_model()

        # Initialize MediaPipe Gesture Recognizer Task
        base_options = python.BaseOptions(model_asset_path=MODEL_NAME)
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(options)

        # Controls Layout
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side="top", fill="x", padx=10, pady=10)

        self.btn_start = ttk.Button(control_frame, text="▶️ Start Camera", command=self.start_camera)
        self.btn_start.pack(side="left", padx=5)

        self.btn_stop = ttk.Button(control_frame, text="⏹️ Stop Camera", command=self.stop_camera, state="disabled")
        self.btn_stop.pack(side="left", padx=5)

        self.status_lbl = ttk.Label(control_frame, text="Status: Offline", font=("Helvetica", 10))
        self.status_lbl.pack(side="left", padx=15)

        # Video Frame Canvas
        self.video_canvas = tk.Label(self.root, text="Camera Offline", bg="#1e272e", fg="white")
        self.video_canvas.pack(fill="both", expand=True, padx=10, pady=5)

        # Output Analytics
        out_frame = ttk.LabelFrame(self.root, text=" Recognized Sign / Gesture ")
        out_frame.pack(side="bottom", fill="x", padx=10, pady=10)

        self.metrics_lbl = ttk.Label(out_frame, text="Waiting for camera feed...", font=("Helvetica", 11, "bold"))
        self.metrics_lbl.pack(anchor="w", padx=10, pady=10)

        self.cap = None
        self.is_running = False

    def process_frame(self, frame):
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Run Gesture Recognizer Task
        result = self.recognizer.recognize(mp_image)

        detected_gestures = []

        if result and result.hand_landmarks:
            for i, hand_landmarks in enumerate(result.hand_landmarks):
                # 1. Draw Skeleton Joints
                for lm in hand_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

                # 2. Get Recognized Sign / Gesture Category
                gesture_name = "Unrecognized Sign"
                score = 0.0
                if result.gestures and len(result.gestures) > i:
                    top_gesture = result.gestures[i][0]
                    gesture_name = top_gesture.category_name
                    score = top_gesture.score

                handedness = result.handedness[i][0].category_name if result.handedness else "Hand"
                
                if gesture_name != "None":
                    detected_gestures.append(f"{handedness}: {gesture_name} ({int(score * 100)}%)")

                # Overlay gesture text above hand wrist
                wrist_x = int(hand_landmarks[0].x * w)
                wrist_y = int(hand_landmarks[0].y * h)
                cv2.putText(
                    frame,
                    f"{gesture_name}",
                    (max(10, wrist_x - 30), max(30, wrist_y - 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )

        return frame, detected_gestures

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not access webcam.")
            return

        self.is_running = True
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.status_lbl.config(text="Status: Live Stream Active")
        self.update_loop()

    def update_loop(self):
        if not self.is_running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            processed_frame, gestures = self.process_frame(frame)

            # Render image in GUI
            rgb_display = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(rgb_display)
            img_pil.thumbnail((750, 480))
            self.tk_img = ImageTk.PhotoImage(img_pil)
            self.video_canvas.config(image=self.tk_img, text="")

            # Update Metrics Display
            summary = " | ".join(gestures) if gestures else "No gestures detected"
            self.metrics_lbl.config(text=f"🤟 Detected Signs: {summary}")

        self.root.after(30, self.update_loop)

    def stop_camera(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None

        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.status_lbl.config(text="Status: Offline")
        self.video_canvas.config(image="", text="Camera Offline")

    def __del__(self):
        if hasattr(self, 'recognizer'):
            self.recognizer.close()

if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageVisionApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop_camera(), root.destroy()))
    root.mainloop()