"""Real-time object detection desktop app using YOLO, OpenCV, Tkinter, and Python.

The application supports webcam-based detection, image upload, Persian object labels,
bounding boxes, and Excel export of detection logs.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import pandas as pd
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox, ttk
from ultralytics import YOLO

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# The model file is intentionally not committed to GitHub.
# Ultralytics can download known models such as yolov8s.pt on first run.
MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolov8s.pt")
CONFIDENCE_THRESHOLD = 0.40
WEBCAM_INDEX = 0

FA_TRANSLATIONS = {
    "person": "انسان",
    "bicycle": "دوچرخه",
    "car": "ماشین",
    "motorcycle": "موتورسیکلت",
    "airplane": "هواپیما",
    "bus": "اتوبوس",
    "train": "قطار",
    "truck": "کامیون",
    "boat": "قایق",
    "traffic light": "چراغ راهنما",
    "bench": "نیمکت",
    "bird": "پرنده",
    "cat": "گربه",
    "dog": "سگ",
    "horse": "اسب",
    "sheep": "گوسفند",
    "cow": "گاو",
    "elephant": "فیل",
    "bear": "خرس",
    "zebra": "گورخر",
    "giraffe": "زرافه",
    "backpack": "کوله‌پشتی",
    "umbrella": "چتر",
    "handbag": "کیف‌دستی",
    "tie": "کراوات",
    "suitcase": "چمدان",
    "bottle": "بطری",
    "cup": "لیوان",
    "fork": "چنگال",
    "knife": "چاقو",
    "spoon": "قاشق",
    "bowl": "کاسه",
    "banana": "موز",
    "apple": "سیب",
    "orange": "پرتقال",
    "pizza": "پیتزا",
    "cake": "کیک",
    "chair": "صندلی",
    "sofa": "مبل",
    "potted plant": "گیاه گلدانی",
    "bed": "تخت",
    "dining table": "میز غذاخوری",
    "tv": "تلویزیون",
    "laptop": "لپ‌تاپ",
    "mouse": "ماوس",
    "remote": "ریموت",
    "keyboard": "کیبورد",
    "cell phone": "موبایل",
    "book": "کتاب",
    "clock": "ساعت",
    "vase": "گلدان",
    "scissors": "قیچی",
    "teddy bear": "خرس عروسکی",
    "hair drier": "سشوار",
    "toothbrush": "مسواک",
}


def to_farsi(label_en: str) -> str:
    """Return the Persian label if available; otherwise return the English label."""
    return FA_TRANSLATIONS.get(label_en, label_en)


def sorted_labels(labels: Iterable[str]) -> tuple[list[str], list[str]]:
    """Return sorted English labels and their Persian equivalents."""
    english_labels = sorted(labels)
    persian_labels = [to_farsi(label) for label in english_labels]
    return english_labels, persian_labels


class ObjectDetectionApp:
    """Tkinter desktop application for real-time and image-based object detection."""

    def __init__(self) -> None:
        self.model = YOLO(MODEL_PATH)
        self.current_objects_en: set[str] = set()
        self.detections_log: list[dict[str, str]] = []
        self.cap = cv2.VideoCapture(WEBCAM_INDEX)

        self.root = tk.Tk()
        self.root.title("سیستم تشخیص اشیا با YOLO")
        self.root.configure(bg="#020617")
        self.root.geometry("1150x650")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self._configure_style()
        self._build_ui()

    def _configure_style(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#020617")
        style.configure(
            "Title.TLabel",
            background="#020617",
            foreground="white",
            font=("Tahoma", 14, "bold"),
        )
        style.configure(
            "TLabel",
            background="#020617",
            foreground="white",
            font=("Tahoma", 11),
        )
        style.configure("TButton", font=("Tahoma", 11), padding=6)
        style.map(
            "TButton",
            foreground=[("active", "white")],
            background=[("active", "#22c55e")],
        )

    def _build_ui(self) -> None:
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=12, pady=12)

        video_frame = ttk.Frame(main_frame)
        video_frame.pack(side="left", fill="both", expand=True)

        title_label = ttk.Label(video_frame, text="نمای زنده وبکم", style="Title.TLabel")
        title_label.pack(anchor="w", pady=(0, 6))

        self.video_label = ttk.Label(video_frame, text="در حال بارگذاری وبکم...")
        self.video_label.pack(fill="both", expand=True)

        side_frame = ttk.Frame(main_frame)
        side_frame.pack(side="right", fill="y", padx=(10, 0))

        list_title = ttk.Label(side_frame, text="اشیای شناسایی‌شده", style="Title.TLabel")
        list_title.pack(anchor="center", pady=(0, 8))

        list_frame = ttk.Frame(side_frame)
        list_frame.pack(fill="both", expand=True)

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")

        self.object_listbox = tk.Listbox(
            list_frame,
            yscrollcommand=scrollbar.set,
            bg="#020617",
            fg="white",
            font=("Tahoma", 11),
            highlightthickness=0,
            borderwidth=1,
            relief="solid",
            selectbackground="#22c55e",
            activestyle="none",
        )
        self.object_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.object_listbox.yview)

        btn_frame = ttk.Frame(side_frame)
        btn_frame.pack(fill="x", pady=10)

        download_button = ttk.Button(btn_frame, text="ذخیره اکسل", command=self.save_to_excel)
        download_button.pack(fill="x", pady=3)

        upload_button = ttk.Button(btn_frame, text="آپلود عکس و تشخیص", command=self.upload_image)
        upload_button.pack(fill="x", pady=3)

        exit_button = ttk.Button(btn_frame, text="خروج", command=self.on_closing)
        exit_button.pack(fill="x", pady=3)

    def save_to_excel(self) -> None:
        """Export detection logs to an Excel file."""
        if not self.detections_log:
            messagebox.showinfo("ذخیره اکسل", "هنوز هیچ داده‌ای ثبت نشده است.")
            return

        filename = filedialog.asksaveasfilename(
            title="ذخیره گزارش تشخیص",
            defaultextension=".xlsx",
            initialfile="detections.xlsx",
            filetypes=[("Excel Files", "*.xlsx")],
        )
        if not filename:
            return

        df = pd.DataFrame(self.detections_log)
        df.to_excel(filename, index=False)
        messagebox.showinfo("ذخیره اکسل", f"فایل با موفقیت ذخیره شد:\n{filename}")

    def update_frame(self) -> None:
        """Read webcam frames, run YOLO, and display annotated frames in Tkinter."""
        if not self.cap.isOpened():
            self.video_label.config(text="عدم دسترسی به وبکم")
            return

        ret, frame = self.cap.read()
        if not ret:
            self.video_label.config(text="عدم دسترسی به وبکم")
            return

        results = self.model(frame, verbose=False)[0]
        self.current_objects_en.clear()

        for box in results.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            if confidence < CONFIDENCE_THRESHOLD:
                continue

            label_en = self.model.names[cls_id]
            self.current_objects_en.add(label_en)

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{label_en} {confidence:.2f}"
            cv2.putText(
                frame,
                text,
                (x1, max(y1 - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=image)

        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(30, self.update_frame)

    def log_every_second(self) -> None:
        """Log detected webcam objects every second."""
        if self.current_objects_en:
            now = datetime.now().strftime("%H:%M:%S")
            english_labels, persian_labels = sorted_labels(self.current_objects_en)

            row = {
                "زمان": now,
                "منبع": "وبکم",
                "اشیای شناسایی‌شده (انگلیسی)": ", ".join(english_labels),
                "اشیای شناسایی‌شده (فارسی)": ", ".join(persian_labels),
            }
            self.detections_log.append(row)

            display_text = f"{now} | وبکم | {row['اشیای شناسایی‌شده (فارسی)']}"
            self.object_listbox.insert(0, display_text)

        self.root.after(1000, self.log_every_second)

    def upload_image(self) -> None:
        """Run object detection on an uploaded image and display the result."""
        filepath = filedialog.askopenfilename(
            title="انتخاب تصویر",
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")],
        )
        if not filepath:
            return

        img_bgr = cv2.imread(filepath)
        if img_bgr is None:
            messagebox.showerror("خطا", "خواندن تصویر انجام نشد.")
            return

        results = self.model(img_bgr, verbose=False)[0]
        objects_en: set[str] = set()

        for box in results.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            if confidence < CONFIDENCE_THRESHOLD:
                continue

            label_en = self.model.names[cls_id]
            objects_en.add(label_en)

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{label_en} {confidence:.2f}"
            cv2.putText(
                img_bgr,
                text,
                (x1, max(y1 - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        if objects_en:
            now = datetime.now().strftime("%H:%M:%S")
            english_labels, persian_labels = sorted_labels(objects_en)

            row = {
                "زمان": now,
                "منبع": "عکس",
                "اشیای شناسایی‌شده (انگلیسی)": ", ".join(english_labels),
                "اشیای شناسایی‌شده (فارسی)": ", ".join(persian_labels),
            }
            self.detections_log.append(row)

            display_text = f"{now} | عکس | {row['اشیای شناسایی‌شده (فارسی)']}"
            self.object_listbox.insert(0, display_text)
        else:
            messagebox.showinfo("نتیجه", "هیچ شیئی در تصویر شناسایی نشد.")

        self._show_result_image(img_bgr)

    def _show_result_image(self, img_bgr) -> None:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        max_w, max_h = 800, 600
        scale = min(max_w / pil_img.width, max_h / pil_img.height, 1.0)
        if scale < 1.0:
            new_size = (int(pil_img.width * scale), int(pil_img.height * scale))
            pil_img = pil_img.resize(new_size, Image.LANCZOS)

        top = tk.Toplevel(self.root)
        top.title("نتیجه تشخیص روی عکس")
        top.configure(bg="#020617")

        img_tk = ImageTk.PhotoImage(pil_img)
        label = ttk.Label(top, image=img_tk)
        label.image = img_tk
        label.pack(padx=8, pady=8)

    def on_closing(self) -> None:
        """Release resources and close the application safely."""
        try:
            if self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()
        self.root.destroy()

    def run(self) -> None:
        self.update_frame()
        self.log_every_second()
        self.root.mainloop()


if __name__ == "__main__":
    ObjectDetectionApp().run()
