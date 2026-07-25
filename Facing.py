import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import mediapipe as mp
from PIL import Image, ImageTk

class HumanRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Offline Person, Face & Finger Count AI Model")
        self.root.geometry("900x750")

        # Initialize MediaPipe Modules
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        
        self.mp_face = mp.solutions.face_detection
        self.face_detection = self.mp_face.FaceDetection(min_detection_confidence=0.6)
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5)

        self.mp_draw = mp.solutions.drawing_utils

        # Top Controls
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side="top", fill="x", padx=10, pady=10)

        self.btn_start = ttk.Button(control_frame, text="▶️ Start Live Camera", command=self.start_camera)
        self.btn_start.pack(side="left", padx=5)

        self.btn_stop = ttk.Button(control_frame, text="⏹️ Stop Camera", command=self.stop_camera, state="disabled")
        self.btn_stop.pack(side="left", padx=5)

        self.btn_file = ttk.Button(control_frame, text="📁 Process Image File", command=self.process_file)
        self.btn_file.pack(side="left", padx=5)

        self.status_lbl = ttk.Label(control_frame, text="Status: Ready", font=("Helvetica", 10))
        self.status_lbl.pack(side="left", padx=15)

        # Video Output Frame
        self.video_canvas = tk.Label(self.root, text="Camera / Image Display", bg="#1e272e", fg="white")
        self.video_canvas.pack(fill="both", expand=True, padx=10, pady=5)

        # Real-time Metrics Dashboard
        out_frame = ttk.LabelFrame(self.root, text=" Live Recognition Analytics ")
        out_frame.pack(side="bottom", fill="x", padx=10, pady=10)

        self.metrics_lbl = ttk.Label(out_frame, text="Start the camera or load an image to display counts.", font=("Helvetica", 11, "bold"))
        self.metrics_lbl.pack(anchor="w", padx=10, pady=10)

        self.cap = None
        self.is_running = False

    def count_raised_fingers(self, hand_landmarks, handedness_label):
        """Calculates how many fingers are held up on a hand."""
        tip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips
        fingers = []

        # Thumb logic (horizontal position check depending on Left vs Right hand)
        if handedness_label == 'Right':
            fingers.append(1 if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x else 0)
        else:
            fingers.append(1 if hand_landmarks.landmark[tip_ids[0]].x > hand_landmarks.landmark[tip_ids[0] - 1].x else 0)

        # 4 Fingers logic (vertical Y coordinate check against lower joint)
        for tid in range(1, 5):
            if hand_landmarks.landmark[tip_ids[tid]].y < hand_landmarks.landmark[tip_ids[tid] - 2].y:
                fingers.append(1)
            else:
                fingers.append(0)

        return sum(fingers)

    def process_frame(self, frame):
        """Processes an image frame for Person, Face, and Hand Finger counts."""
        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        person_count = 0
        face_count = 0
        total_fingers = 0
        hands_detected = 0

        # 1. Person / Pose Recognition
        pose_res = self.pose.process(rgb_frame)
        if pose_res.pose_landmarks:
            person_count = 1
            self.mp_draw.draw_landmarks(frame, pose_res.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        # 2. Face Recognition
        face_res = self.face_detection.process(rgb_frame)
        if face_res.detections:
            face_count = len(face_res.detections)
            for detection in face_res.detections:
                bbox = detection.location_data.relative_bounding_box
                x, y, bw, bh = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (255, 0, 0), 2)
                cv2.putText(frame, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # 3. Hand & Finger Count Recognition
        hand_res = self.hands.process(rgb_frame)
        if hand_res.multi_hand_landmarks and hand_res.multi_handedness:
            hands_detected = len(hand_res.multi_hand_landmarks)
            for hand_lms, handedness in zip(hand_res.multi_hand_landmarks, hand_res.multi_handedness):
                label = handedness.classification[0].label  # 'Left' or 'Right'
                fingers = self.count_raised_fingers(hand_lms, label)
                total_fingers += fingers

                # Draw Hand Skeleton & Finger Count Label
                self.mp_draw.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)
                cx = int(hand_lms.landmark[0].x * w)
                cy = int(hand_lms.landmark[0].y * h)
                cv2.putText(frame, f"{label}: {fingers} Fingers", (cx - 30, cy + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return frame, person_count, face_count, hands_detected, total_fingers

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Could not access webcam!")
            return

        self.is_running = True
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.status_lbl.config(text="Status: Live Camera Feed")
        self.update_loop()

    def update_loop(self):
        if not self.is_running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if ret:
            processed_frame, person_count, face_count, hands_count, finger_count = self.process_frame(frame)

            # Convert BGR -> RGB for Tkinter GUI Display
            rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(rgb)
            img_pil.thumbnail((750, 480))
            self.tk_img = ImageTk.PhotoImage(img_pil)
            self.video_canvas.config(image=self.tk_img, text="")

            # Update Metrics Box
            metrics_text = (
                f"👤 Persons Detected: {person_count}  |  "
                f"😀 Faces Detected: {face_count}  |  "
                f"✋ Hands: {hands_count}  |  "
                f"🖐️ Total Raised Fingers: {finger_count}"
            )
            self.metrics_lbl.config(text=metrics_text)

        self.root.after(30, self.update_loop)

    def process_file(self):
        self.stop_camera()
        file_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg *.bmp")])
        if not file_path:
            return

        frame = cv2.imread(file_path)
        if frame is None:
            return

        processed_frame, person_count, face_count, hands_count, finger_count = self.process_frame(frame)
        rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb)
        img_pil.thumbnail((750, 480))
        self.tk_img = ImageTk.PhotoImage(img_pil)
        self.video_canvas.config(image=self.tk_img, text="")

        metrics_text = (
            f"👤 Persons Detected: {person_count}  |  "
            f"😀 Faces Detected: {face_count}  |  "
            f"✋ Hands: {hands_count}  |  "
            f"🖐️ Total Raised Fingers: {finger_count}"
        )
        self.metrics_lbl.config(text=metrics_text)
        self.status_lbl.config(text=f"Processed: {file_path.split('/')[-1]}")

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
    app = HumanRecognitionApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop_camera(), root.destroy()))
    root.mainloop()