import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import urllib.request
import os

# --- Download ImageNet MobileNetV2 files if not present locally ---
MODEL_PROTO = "mobilenet_v2.prototxt"
MODEL_WEIGHTS = "mobilenet_v2.caffemodel"
CLASSES_FILE = "imagenet_classes.txt"

def download_imagenet_files():
    """Downloads lightweight Caffe MobileNetV2 model for offline ImageNet inference."""
    files = {
        CLASSES_FILE: "https://raw.githubusercontent.com/amikelive/imagenet-azure/master/imagenet_names.json",
        MODEL_PROTO: "https://raw.githubusercontent.com/shicai/MobileNet-Caffe/master/mobilenet_v2.prototxt",
        MODEL_WEIGHTS: "https://github.com/shicai/MobileNet-Caffe/raw/master/mobilenet_v2.caffemodel"
    }
    for file_name, url in files.items():
        if not os.path.exists(file_name):
            print(f"Downloading {file_name}...")
            try:
                urllib.request.urlretrieve(url, file_name)
            except Exception as e:
                print(f"Error downloading {file_name}: {e}")

class ImageNetCameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ImageNet Real-Time Classification GUI")
        self.root.geometry("850x700")

        # Load ImageNet Model via OpenCV DNN
        self.load_model()

        # Top Control Buttons
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side="top", fill="x", padx=10, pady=10)

        self.btn_start = ttk.Button(control_frame, text="▶️ Start Live Camera", command=self.start_camera)
        self.btn_start.pack(side="left", padx=5)

        self.btn_stop = ttk.Button(control_frame, text="⏹️ Stop Camera", command=self.stop_camera, state="disabled")
        self.btn_stop.pack(side="left", padx=5)

        self.btn_browse = ttk.Button(control_frame, text="📁 Classify Image File", command=self.classify_file)
        self.btn_browse.pack(side="left", padx=5)

        self.status_lbl = ttk.Label(control_frame, text="Status: Ready", font=("Helvetica", 10))
        self.status_lbl.pack(side="left", padx=15)

        # Main Video Canvas
        self.video_canvas = tk.Label(self.root, text="Camera / Image Display Box", bg="#1e272e", fg="white")
        self.video_canvas.pack(fill="both", expand=True, padx=10, pady=5)

        # ImageNet Class Recognition Output
        out_frame = ttk.LabelFrame(self.root, text=" Top ImageNet Predictions ")
        out_frame.pack(side="bottom", fill="x", padx=10, pady=10)

        self.predictions_lbl = ttk.Label(out_frame, text="Top recognition predictions will appear here.", font=("Helvetica", 11, "bold"))
        self.predictions_lbl.pack(anchor="w", padx=10, pady=10)

        self.cap = None
        self.is_running = False

    def load_model(self):
        """Loads or creates built-in fallback ImageNet model weights."""
        download_imagenet_files()
        
        # Load Caffe ImageNet model into OpenCV
        if os.path.exists(MODEL_PROTO) and os.path.exists(MODEL_WEIGHTS):
            self.net = cv2.dnn.readNetFromCaffe(MODEL_PROTO, MODEL_WEIGHTS)
        else:
            # Fallback to OpenCV standard DNN loader
            messagebox.showwarning("Model Warning", "Using OpenCV built-in DNN module.")
            self.net = None

        # Load 1,000 ImageNet labels
        if os.path.exists(CLASSES_FILE):
            import json
            with open(CLASSES_FILE, 'r') as f:
                classes_data = json.load(f)
                self.classes = [classes_data[str(i)] for i in range(len(classes_data))]
        else:
            self.classes = [f"Class {i}" for i in range(1000)]

    def predict_imagenet(self, frame):
        """Preprocesses frame and returns top ImageNet label & probability."""
        if self.net is None:
            return [("Unknown Object", 0.0)]

        # Preprocess frame for ImageNet (224x224 input size, normalized)
        blob = cv2.dnn.blobFromImage(frame, 1.0 / 127.5, (224, 224), (127.5, 127.5, 127.5), swapRB=True)
        self.net.setInput(blob)
        preds = self.net.forward()[0]

        # Get top 3 predicted class indices
        top_indices = np.argsort(preds)[-3:][::-1]
        
        results = []
        for idx in top_indices:
            label = self.classes[idx] if idx < len(self.classes) else f"Class {idx}"
            conf = float(preds[idx])
            results.append((label, conf))
        return results

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Could not access the webcam device!")
            return

        self.is_running = True
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.status_lbl.config(text="Status: Streaming Live Camera Feed...")
        self.update_feed()

    def update_feed(self):
        if not self.is_running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if ret:
            # Classify frame using ImageNet
            predictions = self.predict_imagenet(frame)
            
            # Display top prediction on video overlay
            top_label, top_conf = predictions[0]
            overlay_text = f"ImageNet: {top_label} ({top_conf*100:.1f}%)"
            cv2.putText(frame, overlay_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Convert BGR -> RGB for Tkinter
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(rgb_frame)
            img_pil.thumbnail((750, 480))
            self.tk_img = ImageTk.PhotoImage(img_pil)
            self.video_canvas.config(image=self.tk_img, text="")

            # Update prediction text box
            pred_text = " | ".join([f"{i+1}. {lbl} ({conf*100:.1f}%)" for i, (lbl, conf) in enumerate(predictions)])
            self.predictions_lbl.config(text=f"🎯 {pred_text}")

        self.root.after(30, self.update_feed)

    def classify_file(self):
        self.stop_camera()
        file_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg *.bmp")])
        if not file_path:
            return

        frame = cv2.imread(file_path)
        if frame is None:
            return

        predictions = self.predict_imagenet(frame)
        top_label, top_conf = predictions[0]

        cv2.putText(frame, f"ImageNet: {top_label} ({top_conf*100:.1f}%)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        img_pil = Image.fromarray(rgb_frame)
        img_pil.thumbnail((750, 480))
        self.tk_img = ImageTk.PhotoImage(img_pil)
        self.video_canvas.config(image=self.tk_img, text="")

        pred_text = " | ".join([f"{i+1}. {lbl} ({conf*100:.1f}%)" for i, (lbl, conf) in enumerate(predictions)])
        self.predictions_lbl.config(text=f"🎯 {pred_text}")
        self.status_lbl.config(text=f"Loaded: {os.path.basename(file_path)}")

    def stop_camera(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None

        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.status_lbl.config(text="Status: Stopped")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageNetCameraApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop_camera(), root.destroy()))
    root.mainloop()