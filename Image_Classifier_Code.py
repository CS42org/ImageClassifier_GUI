import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from sklearn.model_selection import train_test_split
import random
import tkfilebrowser
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class TrainingProgressCallback(Callback):
    def __init__(self, total_epochs, label_widget, history_tracker):
        super().__init__()
        self.total_epochs = total_epochs
        self.label_widget = label_widget
        self.history_tracker = history_tracker

    def on_epoch_end(self, epoch, logs=None):
        self.label_widget.config(text=f"✅ Epoch {epoch+1}/{self.total_epochs} Complete")
        self.label_widget.update()
        if logs:
            self.history_tracker['accuracy'].append(logs.get('accuracy'))
            self.history_tracker['val_accuracy'].append(logs.get('val_accuracy'))

class ImageClassifierUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🧠 Image Classifier Trainer")
        self.root.geometry("800x1050")

        self.is_dark_mode = True
        self.img_size = tk.IntVar(value=50)
        self.epochs = tk.IntVar(value=25)
        self.test_split_percent = tk.IntVar(value=20)

        self.class_folders = []
        self.X_val = None
        self.y_val = None
        self.class_names = []
        self.model = None
        self.train_history = None
        self.history_tracker = {
            'accuracy': [],
            'val_accuracy': []
        }

        self.accuracy_label = None
        self.setup_ui()
        self.toggle_dark_mode(force=True)

    def setup_ui(self):
        self.root.configure(padx=20, pady=20)

        title = tk.Label(self.root, text="🧠 Image Classifier Trainer", font=("Helvetica", 18, "bold"))
        title.pack(pady=10)

        credit = tk.Label(self.root, text="By CS42.org", font=("Helvetica", 9, "italic"))
        credit.pack(pady=(0, 10))

        frame_inputs = tk.Frame(self.root)
        frame_inputs.pack(pady=10, fill=tk.X)

        tk.Label(frame_inputs, text="📅 Image Size:").grid(row=0, column=0, sticky="w")
        tk.Entry(frame_inputs, textvariable=self.img_size, width=10).grid(row=0, column=1)

        tk.Label(frame_inputs, text="📅 Epochs:").grid(row=1, column=0, sticky="w")
        tk.Entry(frame_inputs, textvariable=self.epochs, width=10).grid(row=1, column=1)

        tk.Label(frame_inputs, text="📊 Test %:").grid(row=2, column=0, sticky="w")
        tk.Scale(frame_inputs, from_=5, to=50, orient=tk.HORIZONTAL, variable=self.test_split_percent).grid(row=2, column=1)

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=15)

        def styled_button(master, text, command, row, col):
            btn = tk.Button(master, text=text, command=command, font=("Helvetica", 10), width=20, relief=tk.GROOVE, bd=2)
            btn.grid(row=row, column=col, padx=8, pady=6)

        styled_button(btn_frame, "📂 Select Class Folders", self.select_multiple_class_folders, 0, 0)
        styled_button(btn_frame, "🚀 Start Training", self.start_training, 0, 1)
        styled_button(btn_frame, "📥 Load Model", self.load_trained_model, 0, 2)
        styled_button(btn_frame, "🔍 Predict Image", self.predict_image_from_file, 1, 0)
        styled_button(btn_frame, "📈 Show Training History", self.show_training_history, 1, 1)
        styled_button(btn_frame, "🪩 Reset & Stop", self.reset_ui, 1, 2)
        styled_button(btn_frame, "🌓 Toggle Theme", self.toggle_dark_mode, 2, 1)

        self.class_listbox = tk.Listbox(self.root, width=80, height=5)
        self.class_listbox.pack(pady=10)

        self.progress_label = tk.Label(self.root, text="")
        self.progress_label.pack(pady=5)

        self.accuracy_label = tk.Label(self.root, text="", font=("Helvetica", 11))
        self.accuracy_label.pack(pady=5)

        self.image_label = tk.Label(self.root, width=200, height=200, relief=tk.SOLID, bd=1)
        self.image_label.pack(pady=10)

        self.prediction_label = tk.Label(self.root, text="", font=("Helvetica", 12, "bold"), relief=tk.SOLID, bd=1, padx=10, pady=5)
        self.prediction_label.pack(pady=10)

        self.samples_frame = tk.Frame(self.root)
        self.samples_frame.pack(pady=10)

    def toggle_dark_mode(self, force=None):
        if force is not None:
            self.is_dark_mode = force
        else:
            self.is_dark_mode = not self.is_dark_mode

        bg = "#1e1e1e" if self.is_dark_mode else "#f0f2f5"
        fg = "white" if self.is_dark_mode else "#333"
        entry_bg = "#2b2b2b" if self.is_dark_mode else "white"

        self.root.configure(bg=bg)
        for widget in self.root.winfo_children():
            try:
                widget.configure(bg=bg, fg=fg)
                for sub in widget.winfo_children():
                    sub.configure(bg=bg, fg=fg)
                    if isinstance(sub, tk.Entry):
                        sub.configure(bg=entry_bg, fg=fg, insertbackground=fg)
            except:
                pass

    def select_multiple_class_folders(self):
        selected_dirs = tkfilebrowser.askopendirnames(title="Select Class Folders", initialdir=os.getcwd())
        if not selected_dirs:
            return
        self.class_folders.clear()
        self.class_listbox.delete(0, tk.END)
        for folder in selected_dirs:
            class_name = os.path.basename(folder)
            self.class_folders.append((class_name, folder))
            self.class_listbox.insert(tk.END, f"{class_name} - {folder}")

    def image_processing(self, image_path):
        try:
            im = cv2.imread(image_path)
            im = cv2.resize(im, (self.img_size.get(), self.img_size.get()))
            return im / 255.0
        except:
            return None

    def load_trained_model(self):
        model_path = filedialog.askopenfilename(title="Load Trained Model", filetypes=[("Keras Model", "*.keras")])
        if model_path:
            self.model = load_model(model_path)
            class_file = model_path.replace('.keras', '_classes.txt')
            if os.path.exists(class_file):
                with open(class_file, 'r') as f:
                    self.class_names = [line.strip() for line in f.readlines()]
            messagebox.showinfo("✅ Loaded", f"Model loaded from:\n{model_path}")

    def predict_image_from_file(self):
        if self.model is None or not self.class_names:
            messagebox.showwarning("⚠️ No Model", "Train or load a model first.")
            return

        file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
        if not file_path:
            return

        image = self.image_processing(file_path)
        if image is None:
            messagebox.showerror("Error", "Could not process image.")
            return

        pred = self.model.predict(image.reshape(1, self.img_size.get(), self.img_size.get(), 3))[0]
        pred_index = np.argmax(pred)
        pred_label = self.class_names[pred_index]
        confidence = pred[pred_index] * 100

        img = Image.fromarray((image * 255).astype(np.uint8)).resize((200, 200))
        img_tk = ImageTk.PhotoImage(img)
        self.image_label.configure(image=img_tk)
        self.image_label.image = img_tk
        self.prediction_label.config(text=f"🤖 {pred_label} ({confidence:.2f}%)")

    def show_sample_predictions(self):
        for widget in self.samples_frame.winfo_children():
            widget.destroy()

        sample_indices = np.random.choice(len(self.X_val), size=min(5, len(self.X_val)), replace=False)
        for idx in sample_indices:
            img = self.X_val[idx]
            true_label = self.class_names[self.y_val[idx]]
            pred = self.model.predict(img.reshape(1, self.img_size.get(), self.img_size.get(), 3))[0]
            pred_label = self.class_names[np.argmax(pred)]

            img_resized = Image.fromarray((img * 255).astype(np.uint8)).resize((100, 100))
            img_tk = ImageTk.PhotoImage(img_resized)
            panel = tk.Label(self.samples_frame, image=img_tk)
            panel.image = img_tk
            panel.pack(side=tk.LEFT, padx=5)

            label = tk.Label(self.samples_frame, text=f"Actual:\n{true_label}\n\nPredicted:\n{pred_label}", font=("Helvetica", 9))
            label.pack(side=tk.LEFT, padx=5)

    def show_training_history(self):
        if not self.history_tracker['accuracy']:
            messagebox.showinfo("Info", "No training history available.")
            return

        top = tk.Toplevel(self.root)
        top.title("📊 Live Accuracy History")
        top.geometry("600x400")

        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)

        def animate(i):
            ax.clear()
            ax.plot(self.history_tracker['accuracy'], label='Accuracy')
            ax.plot(self.history_tracker['val_accuracy'], label='Val Accuracy')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_title('Model Accuracy Over Epochs')
            ax.legend()
            ax.grid(True)

        self._training_anim = FuncAnimation(fig, animate, interval=1000)
        canvas = FigureCanvasTkAgg(fig, master=top)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def reset_ui(self):
        self.class_folders.clear()
        self.class_listbox.delete(0, tk.END)
        self.model = None
        self.train_history = None
        self.class_names = []
        self.image_label.config(image='')
        self.prediction_label.config(text='')
        self.progress_label.config(text='')
        self.accuracy_label.config(text='')
        self.history_tracker = {
            'accuracy': [],
            'val_accuracy': []
        }

    def load_data(self):
        data, labels, class_names = [], [], []
        for idx, (class_name, folder_path) in enumerate(self.class_folders):
            class_names.append(class_name)
            for file in os.listdir(folder_path):
                img_path = os.path.join(folder_path, file)
                if os.path.isfile(img_path):
                    im = self.image_processing(img_path)
                    if im is not None:
                        data.append(im)
                        labels.append(idx)
        combined = list(zip(data, labels))
        if not combined:
            return np.array([]), np.array([]), class_names
        random.shuffle(combined)
        data[:], labels[:] = zip(*combined)
        return np.array(data), np.array(labels), class_names

    def start_training(self):
        if len(self.class_folders) < 2:
            messagebox.showwarning("Warning", "Please select at least two class folders.")
            return

        X, y, self.class_names = self.load_data()
        if len(X) == 0:
            messagebox.showerror("Error", "No valid images found in the selected folders.")
            return
        X = X.reshape(-1, self.img_size.get(), self.img_size.get(), 3).astype('float32')
        X_train, self.X_val, y_train, self.y_val = train_test_split(X, y, test_size=self.test_split_percent.get() / 100.0)

        self.model = Sequential([
            Input(shape=(self.img_size.get(), self.img_size.get(), 3)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(len(self.class_names), activation='softmax')
        ])
        self.model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        self.history_tracker = {'accuracy': [], 'val_accuracy': []}
        progress_callback = TrainingProgressCallback(self.epochs.get(), self.progress_label, self.history_tracker)
        checkpoint = ModelCheckpoint("model_ui.keras", monitor='val_accuracy', save_best_only=True)

        self.train_history = self.model.fit(
            X_train, y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=self.epochs.get(),
            callbacks=[checkpoint, progress_callback],
            verbose=0
        )

        acc = self.model.evaluate(self.X_val, self.y_val, verbose=0)[1] * 100
        self.accuracy_label.config(text=f"🎯 Final Validation Accuracy: {acc:.2f}%")

        with open("model_ui_classes.txt", "w") as f:
            for name in self.class_names:
                f.write(name + "\n")

        messagebox.showinfo("Training Done", f"Training complete!\nModel saved as .keras\nAccuracy: {acc:.2f}%")
        self.show_sample_predictions()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageClassifierUI(root)
    root.mainloop()
