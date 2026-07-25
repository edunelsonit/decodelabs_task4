import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO

class LiveCameraDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Offline Live Camera Person Detection App")
        self.root.geometry("850x700")

        # Load YOLOv8 Model (Works 100% offline once downloaded)
        try:
            self.model = YOLO("yolov8n.pt")
        except Exception as e:
            messagebox.showerror("Model Error", f"Could not load YOLO model: {e}")

        # Control Panel
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side="top", fill="x", padx=10, pady=10)

        self.btn_start = ttk.Button(control_frame, text="▶️ Start Live Camera", command=self.start_camera)
        self.btn_start.pack(side="left", padx=5)

        self.btn_stop = ttk.Button(control_frame, text="⏹️ Stop Camera", command=self.stop_camera, state="disabled")
        self.btn_stop.pack(side="left", padx=5)

        self.status_lbl = ttk.Label(control_frame, text="Status: Camera Stopped", font=("Helvetica", 10))
        self.status_lbl.pack(side="left", padx=15)

        # Video Canvas Feed
        self.video_canvas = tk.Label(self.root, text="Camera Feed Offline", bg="#2c3e50", fg="white")
        self.video_canvas.pack(fill="both", expand=True, padx=10, pady=5)

        # Statistics Output Box
        out_frame = ttk.LabelFrame(self.root, text=" Live Detection Metrics ")
        out_frame.pack(side="bottom", fill="x", padx=10, pady=10)

        self.metrics_lbl = ttk.Label(out_frame, text="Start the camera to view live detection counts.", font=("Helvetica", 10))
        self.metrics_lbl.pack(anchor="w", padx=10, pady=10)

        # Camera state variables
        self.cap = None
        self.is_running = False

    def start_camera(self):
        # Initialize default camera (0 = built-in webcam)
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Could not access the camera hardware!")
            return

        self.is_running = True
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.status_lbl.config(text="Status: Live Streaming...")

        # Begin video loop
        self.update_frame()

    def update_frame(self):
        if not self.is_running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if ret:
            # Perform YOLOv8 object/person recognition on frame
            results = self.model(frame, verbose=False)
            res = results[0]

            # Count people and items
            person_count = 0
            for box in res.boxes:
                cls_id = int(box.cls[0])
                if self.model.names[cls_id] == "person":
                    person_count += 1

            # Render bounding boxes onto OpenCV frame
            annotated_frame = res.plot()
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            # Convert frame for Tkinter GUI display
            img_pil = Image.fromarray(rgb_frame)
            img_pil.thumbnail((750, 480))
            self.tk_img = ImageTk.PhotoImage(img_pil)
            
            self.video_canvas.config(image=self.tk_img, text="")

            # Update live stats
            self.metrics_lbl.config(
                text=f"• Live People Count: {person_count}\n"
                     f"• Total Objects Detected: {len(res.boxes)}"
            )

        # Schedule next frame refresh (~30 FPS)
        self.root.after(30, self.update_frame)

    def stop_camera(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None

        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.status_lbl.config(text="Status: Camera Stopped")
        self.video_canvas.config(image="", text="Camera Feed Offline")

    def __del__(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = LiveCameraDetectorApp(root)
    # Ensure camera resource releases safely on window exit
    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop_camera(), root.destroy()))
    root.mainloop()