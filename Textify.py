import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class OfflineAIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Offline AI Recognition & Model Desktop App")
        self.root.geometry("750x550")
        self.root.resizable(True, True)

        # Style Configuration
        style = ttk.Style()
        style.theme_use('clam')

        # Main Navigation Tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        # --- Tab 1: Offline Text Recognition ---
        self.tab_text = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_text, text="📝 Text Recognition & Sentiment")
        self.setup_text_tab()

        # --- Tab 2: Offline Image Inspection ---
        self.tab_image = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_image, text="🖼️ Image Recognition & Details")
        self.setup_image_tab()

    # -------------------------------------------------------------
    # TAB 1: TEXT PROCESSING LOGIC
    # -------------------------------------------------------------
    def setup_text_tab(self):
        title_lbl = ttk.Label(self.tab_text, text="Offline Text Sentiment & Feature Recognition", font=("Helvetica", 13, "bold"))
        title_lbl.pack(pady=10)

        instruction = ttk.Label(self.tab_text, text="Enter any sentence or Kaggle dataset review text below:")
        instruction.pack(anchor="w", padx=20)

        self.text_input = tk.Text(self.tab_text, height=6, width=70, font=("Helvetica", 10))
        self.text_input.pack(pady=10, padx=20)
        self.text_input.insert("1.0", "This offline desktop model works great and processes data quickly!")

        btn_analyze = ttk.Button(self.tab_text, text="Analyze Text (Offline)", command=self.analyze_text)
        btn_analyze.pack(pady=5)

        # Frame for output
        out_frame = ttk.LabelFrame(self.tab_text, text=" Analysis Results ")
        out_frame.pack(fill="x", padx=20, pady=15)

        self.text_output_lbl = ttk.Label(out_frame, text="Click 'Analyze Text' to display recognition results.", font=("Helvetica", 10))
        self.text_output_lbl.pack(anchor="w", padx=10, pady=10)

    def analyze_text(self):
        user_text = self.text_input.get("1.0", tk.END).strip()
        if not user_text:
            messagebox.showwarning("Input Error", "Please enter text to analyze!")
            return

        # Lexicon-based local offline sentiment logic
        pos_words = ["good", "great", "excellent", "awesome", "fantastic", "happy", "love", "like", "best", "positive"]
        neg_words = ["bad", "terrible", "horrible", "worst", "sad", "hate", "poor", "error", "fail", "negative"]

        words = [w.strip(".,!?").lower() for w in user_text.split()]
        pos_score = sum(1 for w in words if w in pos_words)
        neg_score = sum(1 for w in words if w in neg_words)

        if pos_score > neg_score:
            sentiment = "POSITIVE 😁"
        elif neg_score > pos_score:
            sentiment = "NEGATIVE 😞"
        else:
            sentiment = "NEUTRAL 😐"

        result_str = (
            f"• Sentiment Result: {sentiment}\n"
            f"• Word Count: {len(words)} words\n"
            f"• Positive Matches: {pos_score}\n"
            f"• Negative Matches: {neg_score}"
        )
        self.text_output_lbl.config(text=result_str)

    # -------------------------------------------------------------
    # TAB 2: IMAGE PROCESSING LOGIC
    # -------------------------------------------------------------
    def setup_image_tab(self):
        title_lbl = ttk.Label(self.tab_image, text="Offline Image File Inspection & Properties", font=("Helvetica", 13, "bold"))
        title_lbl.pack(pady=10)

        btn_browse = ttk.Button(self.tab_image, text="📁 Browse Local Image File", command=self.load_image)
        btn_browse.pack(pady=5)

        self.img_preview = ttk.Label(self.tab_image, text="No image selected")
        self.img_preview.pack(pady=10)

        out_frame = ttk.LabelFrame(self.tab_image, text=" Recognized Image Attributes ")
        out_frame.pack(fill="x", padx=20, pady=10)

        self.img_details = ttk.Label(out_frame, text="Select an image to inspect features.", font=("Helvetica", 10))
        self.img_details.pack(anchor="w", padx=10, pady=10)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )
        if file_path:
            try:
                img = Image.open(file_path)
                width, height = img.size
                format_type = img.format
                mode = img.mode

                # Resize image for offline preview box
                preview_img = img.copy()
                preview_img.thumbnail((250, 250))
                self.tk_img = ImageTk.PhotoImage(preview_img)
                self.img_preview.config(image=self.tk_img, text="")

                details = (
                    f"• File Name: {file_path.split('/')[-1]}\n"
                    f"• Dimensions: {width} x {height} Pixels\n"
                    f"• Aspect Ratio: {width/height:.2f}\n"
                    f"• Image Format: {format_type}\n"
                    f"• Color Mode: {mode}"
                )
                self.img_details.config(text=details)
            except Exception as e:
                messagebox.showerror("Error Loading Image", f"Failed to load image:\n{e}")

# -------------------------------------------------------------
# MAIN LOOP
# -------------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = OfflineAIApp(root)
    root.mainloop()