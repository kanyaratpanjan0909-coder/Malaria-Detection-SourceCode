"""
ระบบตรวจจับและวิเคราะห์โรคมาลาเรีย - ฉบับสมบูรณ์
(YOLO ตัดเซลล์ + CNN จำแนกผล + Grad-CAM + แผงสรุปผลเซลล์เดี่ยวด้านขวามือ)
"""

import os
import datetime
import time

import numpy as np
import cv2
import tensorflow as tf

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

try:
    from ultralytics import YOLO
    ULTRALYTICS_OK = True
except ImportError:
    ULTRALYTICS_OK = False

COLOR_MODEL_PATH = "malaria_final_model.h5"
BW_MODEL_PATH = "malaria_final_model_bw.h5"
YOLO_MODEL_PATH = "best.pt"

IMG_SIZE = 128
CLASS_TH = ["ติดเชื้อมาลาเรีย (Parasitized)", "ปกติ ไม่ติดเชื้อ (Uninfected)"]

COLOR_BG = "#eef2f7"
COLOR_CARD = "#ffffff"
COLOR_PRIMARY = "#1e88e5"
COLOR_PRIMARY_DARK = "#125ea3"
COLOR_GREEN = "#43a047"
COLOR_GREEN_DARK = "#2e7d32"
COLOR_RED = "#e53935"
COLOR_RED_DARK = "#b71c1c"
COLOR_GRAY = "#78909c"
COLOR_TEXT = "#263238"
FONT_TH = "Tahoma"

OUTPUT_DIR = "cropped_cells"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def safe_load_keras(path, channels):
    if not os.path.exists(path):
        print(f"⚠️ ไม่พบไฟล์โมเดล: {path}")
        return None
    try:
        m = tf.keras.models.load_model(path)
        print(f"✅ โหลดโมเดล {path} สำเร็จ")
        return m
    except Exception as e:
        print(f"⚠️ โหลดโมเดล {path} ไม่สำเร็จ: {e}")
        return None


color_model = safe_load_keras(COLOR_MODEL_PATH, channels=3)
bw_model = safe_load_keras(BW_MODEL_PATH, channels=1)

yolo_model = None
if ULTRALYTICS_OK:
    if os.path.exists(YOLO_MODEL_PATH):
        try:
            yolo_model = YOLO(YOLO_MODEL_PATH)
            print("✅ โหลดโมเดล YOLO สำเร็จ")
        except Exception as e:
            print(f"⚠️ โหลดโมเดล YOLO ไม่สำเร็จ: {e}")
    else:
        print(f"⚠️ ไม่พบไฟล์โมเดล YOLO: {YOLO_MODEL_PATH}")
else:
    print("⚠️ ไม่พบไลบรารี ultralytics กรุณาติดตั้งด้วย: pip install ultralytics")


def find_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None


_gradcam_model_cache = {}


def _get_gradcam_graph_model(model, last_conv_layer_name):
    cache_key = id(model)
    cached = _gradcam_model_cache.get(cache_key)
    if cached is not None:
        return cached

    input_shape = model.input_shape[1:]
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    conv_output = None
    for layer in model.layers:
        x = layer(x)
        if layer.name == last_conv_layer_name:
            conv_output = x
    if conv_output is None:
        raise ValueError(f"ไม่พบเลเยอร์ {last_conv_layer_name} ในกราฟที่สร้างใหม่")

    grad_model = tf.keras.Model(inputs, [conv_output, x])
    _gradcam_model_cache[cache_key] = grad_model
    return grad_model


def make_gradcam_heatmap(img_array, model):
    last_conv_layer_name = find_last_conv_layer_name(model)
    if not last_conv_layer_name:
        raise ValueError("ไม่พบเลเยอร์ Conv2D ในโมเดลนี้")

    grad_model = _get_gradcam_graph_model(model, last_conv_layer_name)

    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    max_val = tf.math.reduce_max(heatmap)
    if max_val == 0:
        max_val = 1e-8
    heatmap = tf.maximum(heatmap, 0) / max_val
    return heatmap.numpy()


def overlay_heatmap(img_bgr, heatmap, alpha=0.45, size=(340, 340)):
    base = cv2.resize(img_bgr, size)
    hm = cv2.resize(heatmap, size)
    hm = np.uint8(255 * hm)
    hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    return cv2.addWeighted(base, 1 - alpha, hm_color, alpha, 0)


def build_infection_heatmap_image(base_img_bgr, infected_cells):
    composite = cv2.convertScaleAbs(base_img_bgr, alpha=0.55, beta=35)
    for cell in infected_cells:
        x1, y1, x2, y2 = cell["bbox"]
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue
        try:
            model = color_model if cell["mode"] == "color" else bw_model
            img_input = prepare_input(cell["crop"], cell["mode"])
            heatmap = make_gradcam_heatmap(img_input, model)
            hm = cv2.resize(heatmap, (w, h))
            hm_u8 = np.uint8(255 * hm)
            hm_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
            region = composite[y1:y2, x1:x2]
            composite[y1:y2, x1:x2] = cv2.addWeighted(region, 0.35, hm_color, 0.65, 0)
            
            cv2.rectangle(composite, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.putText(composite, f"#{cell['idx']}", (x1, max(10, y1 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 0, 255), 1, cv2.LINE_AA)
        except Exception as e:
            print(f"⚠️ Grad-CAM แผนที่ความร้อนล้มเหลว เซลล์ #{cell['idx']}: {e}")
            continue
    return composite


def prepare_input(img_bgr, mode):
    if mode == "color":
        resized = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
        img_input = np.expand_dims(resized / 255.0, axis=0)
    else:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
        img_input = np.expand_dims(resized / 255.0, axis=-1)
        img_input = np.expand_dims(img_input, axis=0)
    return img_input


def classify_cell(img_bgr, mode):
    model = color_model if mode == "color" else bw_model
    if model is None:
        raise RuntimeError(f"ยังไม่ได้โหลดโมเดล{'สี' if mode == 'color' else 'ขาวดำ'} กรุณาตรวจสอบไฟล์ .h5")
    img_input = prepare_input(img_bgr, mode)
    preds = model.predict(img_input, verbose=0)
    class_idx = int(np.argmax(preds[0]))
    confidence = float(preds[0][class_idx] * 100)
    return class_idx, confidence, model, img_input


def cv2_to_tk_thumb(img_bgr, size=92):
    h, w = img_bgr.shape[:2]
    scale = size / max(h, w)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img_bgr, (nw, nh))
    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    x_off, y_off = (size - nw) // 2, (size - nh) // 2
    canvas[y_off:y_off + nh, x_off:x_off + nw] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return ImageTk.PhotoImage(Image.fromarray(rgb))


def imread_unicode(path, flag=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), flag)


def imwrite_unicode(path, img):
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)
    return ok


class ScrollableFrame(tk.Frame):
    def __init__(self, parent, height=230, bg=COLOR_BG, **kwargs):
        super().__init__(parent, bg=bg, **kwargs)
        self.canvas = tk.Canvas(self, height=height, bg=bg, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = tk.Frame(self.canvas, bg=bg)

        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas_window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfig(self.canvas_window, width=e.width))

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.canvas.bind("<Enter>", lambda e: self.canvas.bind_all("<MouseWheel>", self._on_wheel))
        self.canvas.bind("<Leave>", lambda e: self.canvas.unbind_all("<MouseWheel>"))

    def _on_wheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def clear(self):
        for w in self.inner.winfo_children():
            w.destroy()


class ZoomPanCanvas(tk.Canvas):
    def __init__(self, parent, bg="#e0e0e0", **kwargs):
        super().__init__(parent, bg=bg, highlightthickness=0, **kwargs)
        self.img_bgr = None
        self.tk_img = None
        self.image_id = None
        self.zoom_scale = 1.0
        self.min_scale = 0.1
        self.max_scale = 20.0

        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<MouseWheel>", self._on_zoom_win)
        self.bind("<Button-4>", self._on_zoom_linux)
        self.bind("<Button-5>", self._on_zoom_linux)

    def set_image(self, img_bgr):
        self.img_bgr = img_bgr
        self.zoom_scale = 1.0
        self.redraw(reset_view=True)

    def redraw(self, reset_view=False):
        if self.img_bgr is None:
            self.delete("all")
            self.tk_img = None
            return
        h, w = self.img_bgr.shape[:2]
        canvas_w = self.winfo_width()
        canvas_h = self.winfo_height()
        if canvas_w < 10:
            canvas_w = 460
        if canvas_h < 10:
            canvas_h = 460

        if reset_view:
            scale = min(canvas_w / w, canvas_h / h)
            self.zoom_scale = scale

        current_w = max(1, int(w * self.zoom_scale))
        current_h = max(1, int(h * self.zoom_scale))

        resized = cv2.resize(self.img_bgr, (current_w, current_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self.tk_img = ImageTk.PhotoImage(Image.fromarray(rgb))

        self.delete("all")
        self.image_id = self.create_image(0, 0, anchor="nw", image=self.tk_img)

    def _on_press(self, event):
        self.scan_mark(event.x, event.y)

    def _on_drag(self, event):
        self.scan_dragto(event.x, event.y, gain=1)

    def _on_zoom_win(self, event):
        if self.img_bgr is None:
            return
        if event.delta > 0:
            self.zoom_scale = min(self.max_scale, self.zoom_scale * 1.15)
        else:
            self.zoom_scale = max(self.min_scale, self.zoom_scale / 1.15)
        self.redraw(reset_view=False)

    def _on_zoom_linux(self, event):
        if self.img_bgr is None:
            return
        if event.num == 4:
            self.zoom_scale = min(self.max_scale, self.zoom_scale * 1.15)
        else:
            self.zoom_scale = max(self.min_scale, self.zoom_scale / 1.15)
        self.redraw(reset_view=False)


class MalariaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ระบบตรวจจับและวิเคราะห์โรคมาลาเรีย (YOLO + CNN + Grad-CAM)")
        self.root.configure(bg=COLOR_BG)
        self.root.geometry("1360x900")
        self.root.minsize(1180, 760)
        try:
            self.root.state("zoomed")
        except tk.TclError:
            pass

        self._setup_style()

        self.mode_var1 = tk.StringVar(value="color")
        self.single_items = []
        self.single_selected_idx = None
        self.single_selector_var = tk.StringVar(value="")

        self.mode_var2 = tk.StringVar(value="color")
        self.smear_items = []
        self.smear_selected_idx = None
        self.gallery_filter_var = tk.StringVar(value="all")
        self.smear_view_var = tk.StringVar(value="boxes")
        self.image_selector_var = tk.StringVar(value="")

        self._build_header()

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=14, pady=(6, 10))

        self.tab_single = tk.Frame(self.notebook, bg=COLOR_BG)
        self.tab_batch = tk.Frame(self.notebook, bg=COLOR_BG)
        self.notebook.add(self.tab_single, text="  🔬  วิเคราะห์เซลล์เดี่ยว  ")
        self.notebook.add(self.tab_batch, text="  🩸  วิเคราะห์แผ่นฟิล์มเลือด  ")

        self._build_single_tab()
        self._build_batch_tab()
        self._build_status_bar()

    def _setup_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TNotebook", background=COLOR_BG, borderwidth=0)
        style.configure("TNotebook.Tab", font=(FONT_TH, 12, "bold"), padding=[18, 10])
        style.map("TNotebook.Tab",
                  background=[("selected", COLOR_PRIMARY)],
                  foreground=[("selected", "white")])

    def _build_header(self):
        header = tk.Frame(self.root, bg=COLOR_PRIMARY, height=64)
        header.pack(fill="x")
        header.pack_propagate(False)
        tk.Label(header, text="🧬  ระบบตรวจจับและวิเคราะห์โรคมาลาเรีย",
                 font=(FONT_TH, 19, "bold"), bg=COLOR_PRIMARY, fg="white").pack(side="left", padx=20)
        tk.Label(header, text="YOLO ตัดเซลล์  +  CNN จำแนกผล  +  Grad-CAM อธิบายผล",
                 font=(FONT_TH, 11), bg=COLOR_PRIMARY, fg="#dceeff").pack(side="left", padx=6)

    def _build_status_bar(self):
        bar = tk.Frame(self.root, bg="#dfe7ee", height=26)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)
        parts = []
        parts.append(("โมเดลสี ✅" if color_model else "โมเดลสี ❌"))
        parts.append(("โมเดลขาวดำ ✅" if bw_model else "โมเดลขาวดำ ❌"))
        parts.append(("YOLO ✅" if yolo_model else "YOLO ❌"))
        tk.Label(bar, text="  |  ".join(parts), font=(FONT_TH, 9), bg="#dfe7ee", fg="#455a64").pack(side="left", padx=12)

    # ========================================================
    # TAB 1: วิเคราะห์เซลล์เดี่ยว
    # ========================================================
    def _build_single_tab(self):
        root_ = self.tab_single

        ctrl = tk.Frame(root_, bg=COLOR_CARD, padx=14, pady=10)
        ctrl.pack(fill="x", padx=10, pady=10)

        tk.Label(ctrl, text="เลือกโหมดโมเดล:", font=(FONT_TH, 11, "bold"), bg=COLOR_CARD, fg=COLOR_TEXT).pack(side="left")
        tk.Radiobutton(ctrl, text="ภาพสี (Color)", variable=self.mode_var1, value="color",
                       font=(FONT_TH, 10), bg=COLOR_CARD).pack(side="left", padx=8)
        tk.Radiobutton(ctrl, text="ภาพขาวดำ (Grayscale)", variable=self.mode_var1, value="bw",
                       font=(FONT_TH, 10), bg=COLOR_CARD).pack(side="left", padx=8)

        tk.Button(ctrl, text="📂 อัปโหลดภาพเซลล์ (เลือกทีละหลายไฟล์ได้)", command=self.upload_single,
                  font=(FONT_TH, 10, "bold"), bg=COLOR_GREEN, fg="white", relief="flat",
                  activebackground=COLOR_GREEN_DARK, padx=12, pady=6).pack(side="right", padx=4)
        tk.Button(ctrl, text="🔍 รันประมวลผล + Grad-CAM ทั้งหมด", command=self.run_single,
                  font=(FONT_TH, 10, "bold"), bg=COLOR_PRIMARY, fg="white", relief="flat",
                  activebackground=COLOR_PRIMARY_DARK, padx=12, pady=6).pack(side="right", padx=4)
        tk.Button(ctrl, text="🗑️ เคลียร์ทั้งหมด", command=self.clear_single,
                  font=(FONT_TH, 10, "bold"), bg=COLOR_GRAY, fg="white", relief="flat",
                  activebackground="#546e7a", padx=12, pady=6).pack(side="right", padx=4)

        selector_bar1 = tk.Frame(root_, bg=COLOR_CARD, padx=14, pady=6)
        selector_bar1.pack(fill="x", padx=10)
        tk.Label(selector_bar1, text="📑 ภาพเซลล์ที่กำลังดู:", font=(FONT_TH, 10, "bold"),
                 bg=COLOR_CARD, fg=COLOR_TEXT).pack(side="left")
        self.single_selector_combo = ttk.Combobox(selector_bar1, textvariable=self.single_selector_var,
                                                  state="readonly", font=(FONT_TH, 9), width=55)
        self.single_selector_combo.pack(side="left", padx=8)
        self.single_selector_combo.bind("<<ComboboxSelected>>", self._on_single_selected)
        tk.Button(selector_bar1, text="◀ ก่อนหน้า", command=self._select_prev_single,
                  font=(FONT_TH, 9), bg=COLOR_GRAY, fg="white", relief="flat", padx=8).pack(side="left", padx=3)
        tk.Button(selector_bar1, text="ถัดไป ▶", command=self._select_next_single,
                  font=(FONT_TH, 9), bg=COLOR_GRAY, fg="white", relief="flat", padx=8).pack(side="left", padx=3)
        
        self.result_banner1 = tk.Frame(root_, bg=COLOR_GRAY, padx=10, pady=14)
        self.result_banner1.pack(fill="x", padx=10, pady=(6, 10))
        self.result_text1 = tk.Label(self.result_banner1, text="สถานะ: พร้อมใช้งาน — กรุณาอัปโหลดรูปภาพเซลล์",
                                   font=(FONT_TH, 16, "bold"), bg=COLOR_GRAY, fg="white", wraplength=1000, justify="center")
        self.result_text1.pack()
        self.result_sub1 = tk.Label(self.result_banner1, text="", font=(FONT_TH, 11), bg=COLOR_GRAY, fg="white", wraplength=1000, justify="center")
        self.result_sub1.pack()

        imgs = tk.Frame(root_, bg=COLOR_BG)
        imgs.pack(fill="both", expand=True, padx=10)

        left = tk.Frame(imgs, bg=COLOR_CARD, padx=10, pady=10)
        left.pack(side="left", fill="both", expand=True, padx=(0, 6))
        
        ltop1 = tk.Frame(left, bg=COLOR_CARD)
        ltop1.pack(fill="x", pady=(0, 6))
        tk.Label(ltop1, text="ภาพเซลล์ต้นฉบับ", font=(FONT_TH, 11, "bold"), bg=COLOR_CARD, fg=COLOR_TEXT).pack(side="left")
        tk.Button(ltop1, text="💾 บันทึกรูปภาพ", command=lambda: self.save_canvas_image(self.canvas_original1),
                  font=(FONT_TH, 8, "bold"), bg=COLOR_GREEN, fg="white", relief="flat", padx=6, pady=2).pack(side="right")
        
        self.canvas_original1 = ZoomPanCanvas(left, bg="#e8ecef", width=380, height=360)
        self.canvas_original1.pack(fill="both", expand=True)
        
        self.single_filename_lbl = tk.Label(left, text="", font=(FONT_TH, 9), bg=COLOR_CARD, fg="#607d8b")
        self.single_filename_lbl.pack(pady=(6, 0))

        mid = tk.Frame(imgs, bg=COLOR_CARD, padx=10, pady=10)
        mid.pack(side="left", fill="both", expand=True, padx=6)
        
        mtop1 = tk.Frame(mid, bg=COLOR_CARD)
        mtop1.pack(fill="x", pady=(0, 6))
        tk.Label(mtop1, text="ผลลัพธ์ Grad-CAM Heatmap", font=(FONT_TH, 11, "bold"), bg=COLOR_CARD, fg=COLOR_TEXT).pack(side="left")
        tk.Button(mtop1, text="💾 บันทึกรูปภาพ", command=lambda: self.save_canvas_image(self.canvas_gradcam1),
                  font=(FONT_TH, 8, "bold"), bg=COLOR_GREEN, fg="white", relief="flat", padx=6, pady=2).pack(side="right")
        
        self.canvas_gradcam1 = ZoomPanCanvas(mid, bg="#e8ecef", width=380, height=360)
        self.canvas_gradcam1.pack(fill="both", expand=True)

        # --- แผงสรุปผลรวมด้านขวามือ (เลียนแบบแท็บที่ 2) ---
        right_panel = tk.Frame(imgs, bg=COLOR_CARD, padx=14, pady=14, width=280)
        right_panel.pack(side="left", fill="y", padx=(6, 0))
        right_panel.pack_propagate(False)

        tk.Label(right_panel, text="สรุปผลเซลล์เดี่ยวทั้งหมด", font=(FONT_TH, 12, "bold"), bg=COLOR_CARD, fg=COLOR_TEXT).pack(pady=(0, 10))

        self.stat_single_total = self._make_stat_row(right_panel, "อัปโหลดทั้งหมด", COLOR_TEXT)
        self.stat_single_infected = self._make_stat_row(right_panel, "ติดเชื้อมาลาเรีย", COLOR_RED)
        self.stat_single_normal = self._make_stat_row(right_panel, "ปกติ ไม่ติดเชื้อ", COLOR_GREEN)

        tk.Frame(right_panel, bg="#cfd8dc", height=1).pack(fill="x", pady=10)

        tk.Label(right_panel, text="รายชื่อภาพที่ติดเชื้อ (ดับเบิลคลิกเพื่อดู)", font=(FONT_TH, 9, "bold"),
                 bg=COLOR_CARD, fg=COLOR_RED_DARK, wraplength=260, justify="left").pack(anchor="w")

        single_list_frame = tk.Frame(right_panel, bg=COLOR_CARD)
        single_list_frame.pack(fill="both", expand=True, pady=(4, 0))
        s_scroll = ttk.Scrollbar(single_list_frame, orient="vertical")
        self.single_infected_listbox = tk.Listbox(single_list_frame, font=(FONT_TH, 9), fg=COLOR_RED_DARK,
                                        bg="#fff5f5", relief="flat", highlightthickness=1,
                                        highlightbackground="#ffcdd2", yscrollcommand=s_scroll.set)
        s_scroll.config(command=self.single_infected_listbox.yview)
        self.single_infected_listbox.pack(side="left", fill="both", expand=True)
        s_scroll.pack(side="right", fill="y")
        self.single_infected_listbox.bind("<Double-Button-1>", self._on_single_listbox_click)

        # แถบ Thumbnail
        gallery_card1 = tk.Frame(root_, bg=COLOR_CARD, padx=10, pady=10)
        gallery_card1.pack(fill="both", expand=False, padx=10, pady=(0, 10))
        gtop1 = tk.Frame(gallery_card1, bg=COLOR_CARD)
        gtop1.pack(fill="x")
        tk.Label(gtop1, text="แกลเลอรีภาพ (คลิกที่รูปเพื่อดูภาพขยาย)", font=(FONT_TH, 10, "bold"), bg=COLOR_CARD, fg=COLOR_TEXT).pack(side="left")

        self.single_gallery = ScrollableFrame(gallery_card1, height=130)
        self.single_gallery.pack(fill="both", expand=True, pady=(8, 0))

    def upload_single(self):
        paths = filedialog.askopenfilenames(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")])
        if not paths:
            return
        for path in paths:
            img = imread_unicode(path, cv2.IMREAD_COLOR)
            if img is None:
                continue
            self.single_items.append({
                "path": path, "img_bgr": img, "class_idx": None, "confidence": 0.0, "cam_img": None, "processed": False
            })

        self._refresh_single_selector()
        self.single_selected_idx = len(self.single_items) - 1
        self._sync_single_selector_to_index()
        self._show_selected_single_image()
        self._update_single_summary()
        self._render_single_gallery()

    def clear_single(self):
        self.single_items = []
        self.single_selected_idx = None
        self.single_selector_var.set("")
        self.single_selector_combo["values"] = []
        self.canvas_original1.set_image(None)
        self.canvas_gradcam1.set_image(None)
        self.single_filename_lbl.config(text="")
        self._set_banner1(COLOR_GRAY, "สถานะ: พร้อมใช้งาน — กรุณาอัปโหลดรูปภาพเซลล์", "")
        self._reset_single_summary()
        self.single_gallery.clear()

    def _refresh_single_selector(self):
        values = []
        for i, item in enumerate(self.single_items):
            base_name = os.path.basename(item['path'])
            if item.get("processed"):
                status = "[🦠 ติดเชื้อ]" if item["class_idx"] == 0 else "[✅ ปกติ]"
                values.append(f"#{i + 1} {status} {base_name}")
            else:
                values.append(f"#{i + 1}  {base_name}")
        self.single_selector_combo["values"] = values

    def _sync_single_selector_to_index(self):
        if self.single_selected_idx is None or not self.single_items:
            self.single_selector_var.set("")
            return
        values = self.single_selector_combo["values"]
        if self.single_selected_idx < len(values):
            self.single_selector_var.set(values[self.single_selected_idx])

    def _on_single_selected(self, event=None):
        idx = self.single_selector_combo.current()
        if idx < 0 or idx >= len(self.single_items):
            return
        self.single_selected_idx = idx
        self._show_selected_single_image()

    def _select_prev_single(self):
        if not self.single_items or self.single_selected_idx is None:
            return
        self.single_selected_idx = (self.single_selected_idx - 1) % len(self.single_items)
        self.single_selector_combo.current(self.single_selected_idx)
        self._show_selected_single_image()

    def _select_next_single(self):
        if not self.single_items or self.single_selected_idx is None:
            return
        self.single_selected_idx = (self.single_selected_idx + 1) % len(self.single_items)
        self.single_selector_combo.current(self.single_selected_idx)
        self._show_selected_single_image()

    def _current_single_item(self):
        if self.single_selected_idx is None or not self.single_items:
            return None
        return self.single_items[self.single_selected_idx]

    def _show_selected_single_image(self):
        item = self._current_single_item()
        if item is None:
            return
        self.canvas_original1.set_image(item["img_bgr"])
        self.single_filename_lbl.config(
            text=f"📄 ภาพที่ {self.single_selected_idx + 1}/{len(self.single_items)}: {os.path.basename(item['path'])}"
        )

        if item["processed"]:
            if item["class_idx"] == 0:
                self._set_banner1(COLOR_RED, f"🦠 ติดเชื้อมาลาเรีย (Parasitized)", f"ความมั่นใจ {item['confidence']:.2f}%")
            else:
                self._set_banner1(COLOR_GREEN, f"✅ ปกติ ไม่ติดเชื้อ (Uninfected)", f"ความมั่นใจ {item['confidence']:.2f}%")
            if item["cam_img"] is not None:
                self.canvas_gradcam1.set_image(item["cam_img"])
        else:
            self._set_banner1(COLOR_GRAY, "สถานะ: โหลดรูปภาพสำเร็จ — ยังไม่ได้รันผลวิเคราะห์", "")
            self.canvas_gradcam1.set_image(None)

    def _set_banner1(self, color, text, sub):
        self.result_banner1.config(bg=color)
        self.result_text1.config(bg=color, text=text)
        self.result_sub1.config(bg=color, text=sub)

    def run_single(self):
        if not self.single_items:
            messagebox.showwarning("แจ้งเตือน", "กรุณาอัปโหลดภาพเซลล์เดี่ยวก่อนกดรันครับ")
            return
        mode = self.mode_var1.get()
        start_time = time.time()
        infected_count = 0
        total_count = len(self.single_items)

        for idx, item in enumerate(self.single_items):
            self._set_banner1(COLOR_PRIMARY, f"⏳ กำลังประมวลผลเซลล์ที่ {idx + 1}/{total_count}...", "")
            self.root.update_idletasks()
            try:
                class_idx, confidence, model, img_input = classify_cell(item["img_bgr"], mode)
                heatmap = make_gradcam_heatmap(img_input, model)
                cam_img = overlay_heatmap(item["img_bgr"], heatmap, size=(400, 400))

                item["class_idx"] = class_idx
                item["confidence"] = confidence
                item["cam_img"] = cam_img
                item["processed"] = True
                if class_idx == 0:
                    infected_count += 1
            except Exception as e:
                print(f"⚠️ ข้อผิดพลาดเซลล์ #{idx + 1}: {e}")

        elapsed = time.time() - start_time
        
        self._refresh_single_selector()
        self._sync_single_selector_to_index()
        self._show_selected_single_image()
        self._update_single_summary()
        self._render_single_gallery()
        
        messagebox.showinfo("เสร็จสิ้น", f"ประมวลผลเซลล์เดี่ยวครบ {total_count} ภาพ\nพบติดเชื้อ {infected_count} ภาพ\n⏱️ เวลาที่ใช้: {elapsed:.2f} วินาที")

    def _reset_single_summary(self):
        self.stat_single_total.config(text="0")
        self.stat_single_infected.config(text="0")
        self.stat_single_normal.config(text="0")
        self.single_infected_listbox.delete(0, tk.END)

    def _update_single_summary(self):
        if not self.single_items:
            self._reset_single_summary()
            return
        
        total = len(self.single_items)
        infected_items = [(i, item) for i, item in enumerate(self.single_items) if item.get("processed") and item.get("class_idx") == 0]
        normal_count = sum(1 for item in self.single_items if item.get("processed") and item.get("class_idx") == 1)

        self.stat_single_total.config(text=str(total))
        self.stat_single_infected.config(text=str(len(infected_items)))
        self.stat_single_normal.config(text=str(normal_count))

        self.single_infected_listbox.delete(0, tk.END)
        for idx, item in infected_items:
            filename = os.path.basename(item["path"])
            self.single_infected_listbox.insert(tk.END, f"#{idx+1} {filename} ({item['confidence']:.1f}%)")
        
        if not infected_items and any(item.get("processed") for item in self.single_items):
             self.single_infected_listbox.insert(tk.END, "— ไม่พบเซลล์ติดเชื้อ —")
        elif not any(item.get("processed") for item in self.single_items):
             self.single_infected_listbox.insert(tk.END, "— ยังไม่ได้รันประมวลผล —")

    def _on_single_listbox_click(self, event):
        sel = self.single_infected_listbox.curselection()
        if not sel:
            return
        idx_str = self.single_infected_listbox.get(sel[0])
        try:
            target_idx = int(idx_str.split(" ")[0].replace("#", "")) - 1
            self._jump_to_single(target_idx)
        except:
            pass

    def _render_single_gallery(self):
        self.single_gallery.clear()
        if not self.single_items:
            return

        cols = 10
        for pos, item in enumerate(self.single_items):
            r, c = divmod(pos, cols)
            border_color = COLOR_RED if item.get("processed") and item.get("class_idx") == 0 else \
                           COLOR_GREEN if item.get("processed") and item.get("class_idx") == 1 else COLOR_GRAY

            cell_frame = tk.Frame(self.single_gallery.inner, bg=COLOR_CARD, padx=4, pady=4,
                                  highlightbackground=border_color, highlightthickness=2)
            cell_frame.grid(row=r, column=c, padx=5, pady=5)

            thumb = cv2_to_tk_thumb(item["img_bgr"], 80)
            btn = tk.Button(cell_frame, image=thumb, relief="flat", bd=0,
                            command=lambda idx=pos: self._jump_to_single(idx))
            btn.image = thumb
            btn.pack()

            if item.get("processed"):
                tag = "🦠 ติดเชื้อ" if item["class_idx"] == 0 else "✅ ปกติ"
                fg_color = COLOR_RED if item["class_idx"] == 0 else COLOR_GREEN
            else:
                tag = "รอประมวลผล"
                fg_color = COLOR_GRAY

            tk.Label(cell_frame, text=f"#{pos+1} {tag}", font=(FONT_TH, 8, "bold"), bg=COLOR_CARD, fg=fg_color).pack()

    def _jump_to_single(self, idx):
        self.single_selected_idx = idx
        self.single_selector_combo.current(idx)
        self._show_selected_single_image()


    # ========================================================
    # TAB 2: วิเคราะห์แผ่นฟิล์มเลือด (YOLO) + Zoom / Pan Canvas
    # ========================================================
    def _build_batch_tab(self):
        root_ = self.tab_batch

        ctrl = tk.Frame(root_, bg=COLOR_CARD, padx=14, pady=10)
        ctrl.pack(fill="x", padx=10, pady=10)

        tk.Label(ctrl, text="เลือกโหมดโมเดล:", font=(FONT_TH, 11, "bold"), bg=COLOR_CARD, fg=COLOR_TEXT).pack(side="left")
        tk.Radiobutton(ctrl, text="ภาพสี (Color)", variable=self.mode_var2, value="color",
                       font=(FONT_TH, 10), bg=COLOR_CARD).pack(side="left", padx=8)
        tk.Radiobutton(ctrl, text="ภาพขาวดำ (Grayscale)", variable=self.mode_var2, value="bw",
                       font=(FONT_TH, 10), bg=COLOR_CARD).pack(side="left", padx=8)

        tk.Button(ctrl, text="📂 อัปโหลดภาพแผ่นฟิล์ม (เลือกได้หลายไฟล์)", command=self.upload_smear,
                  font=(FONT_TH, 10, "bold"), bg=COLOR_GREEN, fg="white", relief="flat",
                  activebackground=COLOR_GREEN_DARK, padx=12, pady=6).pack(side="right", padx=4)
        tk.Button(ctrl, text="🔍 ตัดเซลล์ (YOLO) + จำแนกทั้งหมด", command=self.run_smear,
                  font=(FONT_TH, 10, "bold"), bg=COLOR_PRIMARY, fg="white", relief="flat",
                  activebackground=COLOR_PRIMARY_DARK, padx=12, pady=6).pack(side="right", padx=4)
        tk.Button(ctrl, text="🗑️ เคลียร์ทั้งหมด", command=self.clear_smear,
                  font=(FONT_TH, 10, "bold"), bg=COLOR_GRAY, fg="white", relief="flat",
                  activebackground="#546e7a", padx=12, pady=6).pack(side="right", padx=4)

        selector_bar = tk.Frame(root_, bg=COLOR_CARD, padx=14, pady=6)
        selector_bar.pack(fill="x", padx=10)
        tk.Label(selector_bar, text="📑 ภาพแผ่นฟิล์มที่กำลังดู:", font=(FONT_TH, 10, "bold"),
                 bg=COLOR_CARD, fg=COLOR_TEXT).pack(side="left")
        self.image_selector_combo = ttk.Combobox(selector_bar, textvariable=self.image_selector_var,
                                                 state="readonly", font=(FONT_TH, 9), width=55)
        self.image_selector_combo.pack(side="left", padx=8)
        self.image_selector_combo.bind("<<ComboboxSelected>>", self._on_image_selected)
        tk.Button(selector_bar, text="◀ ก่อนหน้า", command=self._select_prev_image,
                  font=(FONT_TH, 9), bg=COLOR_GRAY, fg="white", relief="flat", padx=8).pack(side="left", padx=3)
        tk.Button(selector_bar, text="ถัดไป ▶", command=self._select_next_image,
                  font=(FONT_TH, 9), bg=COLOR_GRAY, fg="white", relief="flat", padx=8).pack(side="left", padx=3)
        self.image_count_lbl = tk.Label(selector_bar, text="ยังไม่มีภาพที่อัปโหลด",
                                         font=(FONT_TH, 9), bg=COLOR_CARD, fg="#607d8b")
        self.image_count_lbl.pack(side="left", padx=10)

        body = tk.Frame(root_, bg=COLOR_BG)
        body.pack(fill="both", expand=True, padx=10)

        left = tk.Frame(body, bg=COLOR_CARD, padx=10, pady=10)
        left.pack(side="left", fill="both", expand=True, padx=(0, 6))

        left_top = tk.Frame(left, bg=COLOR_CARD)
        left_top.pack(fill="x", pady=(0, 6))
        self.smear_view_title = tk.Label(left_top, text="ภาพแผ่นฟิล์มเลือดต้นฉบับ",
                                       font=(FONT_TH, 11, "bold"), bg=COLOR_CARD, fg=COLOR_TEXT)
        self.smear_view_title.pack(side="left")

        tk.Button(left_top, text="💾 บันทึกรูปภาพ", command=lambda: self.save_canvas_image(self.canvas_smear),
                  font=(FONT_TH, 8, "bold"), bg=COLOR_GREEN, fg="white", relief="flat", padx=6, pady=2).pack(side="right", padx=6)

        view_toggle = tk.Frame(left_top, bg=COLOR_CARD)
        view_toggle.pack(side="right")
        tk.Radiobutton(view_toggle, text="🗺️ กรอบ + พิกัด", variable=self.smear_view_var, value="boxes",
                       font=(FONT_TH, 9), bg=COLOR_CARD, command=self._update_smear_view).pack(side="left", padx=3)
        tk.Radiobutton(view_toggle, text="🌡️ แผนที่ความร้อน (Grad-CAM)", variable=self.smear_view_var, value="heatmap",
                       font=(FONT_TH, 9), bg=COLOR_CARD, command=self._update_smear_view).pack(side="left", padx=3)

        self.canvas_smear = ZoomPanCanvas(left, bg="#e8ecef", width=500, height=450)
        self.canvas_smear.pack(fill="both", expand=True)

        self.smear_filename_lbl = tk.Label(left, text="", font=(FONT_TH, 9), bg=COLOR_CARD, fg="#607d8b")
        self.smear_filename_lbl.pack(pady=(6, 0))

        right = tk.Frame(body, bg=COLOR_CARD, padx=14, pady=14, width=280)
        right.pack(side="left", fill="y", padx=(6, 0))
        right.pack_propagate(False)
        tk.Label(right, text="สรุปผลรวมทุกภาพที่วิเคราะห์", font=(FONT_TH, 13, "bold"), bg=COLOR_CARD, fg=COLOR_TEXT).pack(pady=(0, 10))

        self.stat_images = self._make_stat_row(right, "จำนวนภาพที่วิเคราะห์", COLOR_TEXT)
        self.stat_total = self._make_stat_row(right, "จำนวนเซลล์ทั้งหมด", COLOR_TEXT)
        self.stat_infected = self._make_stat_row(right, "เซลล์ติดเชื้อ", COLOR_RED)
        self.stat_uninfected = self._make_stat_row(right, "เซลล์ปกติ", COLOR_GREEN)

        tk.Frame(right, bg="#cfd8dc", height=1).pack(fill="x", pady=10)

        self.big_verdict = tk.Label(right, text="รอผลตรวจ", font=(FONT_TH, 13, "bold"),
                                   bg=COLOR_CARD, fg=COLOR_GRAY, wraplength=260, justify="center")
        self.big_verdict.pack(pady=(4, 4))
        self.rate_label = tk.Label(right, text="", font=(FONT_TH, 10), bg=COLOR_CARD, fg="#607d8b", wraplength=260, justify="center")
        self.rate_label.pack()

        tk.Frame(right, bg="#cfd8dc", height=1).pack(fill="x", pady=10)
        tk.Label(right, text="พิกัดเซลล์ติดเชื้อ (px) — ภาพที่กำลังดู", font=(FONT_TH, 9, "bold"),
                 bg=COLOR_CARD, fg=COLOR_TEXT, wraplength=260, justify="left").pack(anchor="w")

        coord_frame = tk.Frame(right, bg=COLOR_CARD)
        coord_frame.pack(fill="both", expand=True, pady=(4, 0))
        coord_scroll = ttk.Scrollbar(coord_frame, orient="vertical")
        self.coord_listbox = tk.Listbox(coord_frame, font=(FONT_TH, 9), fg=COLOR_RED_DARK,
                                        bg="#fff5f5", relief="flat", highlightthickness=1,
                                        highlightbackground="#ffcdd2", yscrollcommand=coord_scroll.set)
        coord_scroll.config(command=self.coord_listbox.yview)
        self.coord_listbox.pack(side="left", fill="both", expand=True)
        coord_scroll.pack(side="right", fill="y")
        self.coord_listbox.bind("<Double-Button-1>", self._on_coord_listbox_click)
        self.coord_listbox_cells = []

        gallery_card = tk.Frame(root_, bg=COLOR_CARD, padx=10, pady=10)
        gallery_card.pack(fill="both", expand=False, padx=10, pady=10)

        gtop = tk.Frame(gallery_card, bg=COLOR_CARD)
        gtop.pack(fill="x")
        tk.Label(gtop, text="เซลล์ที่ตัดออกมาได้ — ภาพที่กำลังดู (คลิกรูปเพื่อดูรายละเอียด + Grad-CAM)",
                 font=(FONT_TH, 11, "bold"), bg=COLOR_CARD, fg=COLOR_TEXT).pack(side="left")

        filt = tk.Frame(gtop, bg=COLOR_CARD)
        filt.pack(side="right")
        for val, lbl in [("all", "ทั้งหมด"), ("infected", "เฉพาะติดเชื้อ"), ("uninfected", "เฉพาะปกติ")]:
            tk.Radiobutton(filt, text=lbl, variable=self.gallery_filter_var, value=val,
                           font=(FONT_TH, 9), bg=COLOR_CARD, command=self._render_gallery).pack(side="left", padx=4)

        self.gallery = ScrollableFrame(gallery_card, height=220)
        self.gallery.pack(fill="both", expand=True, pady=(8, 0))

    def save_canvas_image(self, canvas_widget):
        if canvas_widget.img_bgr is None:
            messagebox.showwarning("แจ้งเตือน", "ยังไม่มีรูปภาพในจอแสดงผลสำหรับบันทึกครับ")
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg"), ("All Files", "*.*")],
            title="บันทึกรูปภาพผลลัพธ์"
        )
        if file_path:
            try:
                imwrite_unicode(file_path, canvas_widget.img_bgr)
                messagebox.showinfo("สำเร็จ", f"บันทึกรูปภาพเรียบร้อยแล้วที่:\n{file_path}")
            except Exception as e:
                messagebox.showerror("ข้อผิดพลาด", f"ไม่สามารถบันทึกรูปภาพได้: {e}")

    def _make_stat_row(self, parent, label, color):
        row = tk.Frame(parent, bg=COLOR_CARD)
        row.pack(fill="x", pady=4)
        tk.Label(row, text=label, font=(FONT_TH, 10), bg=COLOR_CARD, fg="#607d8b").pack(side="left")
        val = tk.Label(row, text="0", font=(FONT_TH, 14, "bold"), bg=COLOR_CARD, fg=color)
        val.pack(side="right")
        return val

    def _on_coord_listbox_click(self, event):
        sel = self.coord_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        if idx < len(self.coord_listbox_cells):
            self.show_cell_detail(self.coord_listbox_cells[idx])

    def upload_smear(self):
        paths = filedialog.askopenfilenames(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        if not paths:
            return

        added = 0
        for path in paths:
            img = imread_unicode(path, cv2.IMREAD_COLOR)
            if img is None:
                continue
            
            std_width = 800
            h_orig, w_orig = img.shape[:2]
            if w_orig != std_width:
                std_height = int(h_orig * (std_width / w_orig))
                img = cv2.resize(img, (std_width, std_height), interpolation=cv2.INTER_AREA)

            self.smear_items.append({
                "path": path, "img_bgr": img, "cell_results": [],
                "display_boxed": None, "display_heatmap": None,
                "infected": 0, "uninfected": 0, "processed": False,
            })
            added += 1

        if added == 0:
            messagebox.showerror("ข้อผิดพลาด", "ไม่สามารถอ่านไฟล์รูปภาพที่เลือกได้เลย")
            return

        self._refresh_image_selector()
        self.smear_selected_idx = len(self.smear_items) - 1
        self._sync_selector_to_index()
        self._show_selected_smear_image()

    def clear_smear(self):
        self.smear_items = []
        self.smear_selected_idx = None
        self.image_selector_var.set("")
        self.image_selector_combo["values"] = []
        self.image_count_lbl.config(text="ยังไม่มีภาพที่อัปโหลด")
        self.smear_view_var.set("boxes")
        self.canvas_smear.set_image(None)
        self.smear_filename_lbl.config(text="")
        self._reset_summary()
        self.gallery.clear()
        self.coord_listbox.delete(0, tk.END)
        self.coord_listbox_cells = []

    def _refresh_image_selector(self):
        values = [f"#{i + 1}  {os.path.basename(item['path'])}" for i, item in enumerate(self.smear_items)]
        self.image_selector_combo["values"] = values
        self.image_count_lbl.config(text=f"อัปโหลดแล้วทั้งหมด {len(self.smear_items)} ภาพ")

    def _sync_selector_to_index(self):
        if self.smear_selected_idx is None or not self.smear_items:
            self.image_selector_var.set("")
            return
        item = self.smear_items[self.smear_selected_idx]
        self.image_selector_var.set(f"#{self.smear_selected_idx + 1}  {os.path.basename(item['path'])}")

    def _on_image_selected(self, event=None):
        idx = self.image_selector_combo.current()
        if idx < 0 or idx >= len(self.smear_items):
            return
        self.smear_selected_idx = idx
        self._show_selected_smear_image()

    def _select_prev_image(self):
        if not self.smear_items or self.smear_selected_idx is None:
            return
        self.smear_selected_idx = (self.smear_selected_idx - 1) % len(self.smear_items)
        self.image_selector_combo.current(self.smear_selected_idx)
        self._show_selected_smear_image()

    def _select_next_image(self):
        if not self.smear_items or self.smear_selected_idx is None:
            return
        self.smear_selected_idx = (self.smear_selected_idx + 1) % len(self.smear_items)
        self.image_selector_combo.current(self.smear_selected_idx)
        self._show_selected_smear_image()

    def _current_item(self):
        if self.smear_selected_idx is None or not self.smear_items:
            return None
        return self.smear_items[self.smear_selected_idx]

    def _show_selected_smear_image(self):
        item = self._current_item()
        if item is None:
            self.canvas_smear.set_image(None)
            self.smear_filename_lbl.config(text="")
            self.gallery.clear()
            self.coord_listbox.delete(0, tk.END)
            self.coord_listbox_cells = []
            return

        self._update_smear_view()
        self.smear_filename_lbl.config(
            text=f"📄 ภาพที่ {self.smear_selected_idx + 1}/{len(self.smear_items)}: {os.path.basename(item['path'])}"
        )
        self._render_gallery()

        self.coord_listbox.delete(0, tk.END)
        self.coord_listbox_cells = []
        infected_cells = [c for c in item["cell_results"] if c["class_idx"] == 0]
        for cell in infected_cells:
            x1, y1, x2, y2 = cell["bbox"]
            self.coord_listbox.insert(
                tk.END, f"#{cell['idx']}  ({x1},{y1})-({x2},{y2})  {cell['confidence']:.1f}%"
            )
            self.coord_listbox_cells.append(cell)
        if item["processed"] and not infected_cells:
            self.coord_listbox.insert(tk.END, "— ไม่พบเซลล์ติดเชื้อในภาพนี้ —")
        elif not item["processed"]:
            self.coord_listbox.insert(tk.END, "— ยังไม่ได้ประมวลผลภาพนี้ —")

    def _update_smear_view(self):
        item = self._current_item()
        if item is None:
            self.canvas_smear.set_image(None)
            return
        mode = self.smear_view_var.get()
        img = item["display_heatmap"] if mode == "heatmap" else item["display_boxed"]
        if img is None:
            img = item["img_bgr"]
            if mode == "heatmap":
                self.smear_view_title.config(text="แผนที่ความร้อน Grad-CAM (ยังไม่ได้ประมวลผลภาพนี้)")
            else:
                self.smear_view_title.config(text="ภาพแผ่นฟิล์มเลือดต้นฉบับ (ยังไม่ได้ประมวลผล)")
        else:
            if mode == "heatmap":
                self.smear_view_title.config(text="แผนที่ความร้อน Grad-CAM (เฉพาะบริเวณเซลล์ติดเชื้อ)")
            else:
                self.smear_view_title.config(text="ภาพแผ่นฟิล์มเลือดต้นฉบับ (กรอบแดง = ติดเชื้อ, กรอบเขียว = ปกติ)")
        
        self.canvas_smear.set_image(img)

    def _reset_summary(self):
        self.stat_images.config(text="0")
        self.stat_total.config(text="0")
        self.stat_infected.config(text="0")
        self.stat_uninfected.config(text="0")
        self.big_verdict.config(text="รอผลตรวจ", fg=COLOR_GRAY)
        self.rate_label.config(text="")
        self.smear_view_title.config(text="ภาพแผ่นฟิล์มเลือดต้นฉบับ")

    def run_smear(self):
        if not self.smear_items:
            messagebox.showwarning("แจ้งเตือน", "กรุณาอัปโหลดภาพแผ่นฟิล์มเลือดอย่างน้อย 1 ภาพก่อนกดรันครับ")
            return
        if yolo_model is None:
            messagebox.showerror("ข้อผิดพลาด", "ยังไม่ได้โหลดโมเดล YOLO (best.pt)")
            return
        mode = self.mode_var2.get()
        if (mode == "color" and color_model is None) or (mode == "bw" and bw_model is None):
            messagebox.showerror("ข้อผิดพลาด", "ยังไม่ได้โหลดโมเดล CNN สำหรับโหมดที่เลือก")
            return

        start_time = time.time()
        run_dir = os.path.join(OUTPUT_DIR, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(run_dir, exist_ok=True)

        total_images = len(self.smear_items)
        agg_total, agg_infected, agg_uninfected = 0, 0, 0
        images_with_infection = 0

        for img_idx, item in enumerate(self.smear_items):
            self.big_verdict.config(
                text=f"⏳ กำลังประมวลผลภาพที่ {img_idx + 1}/{total_images}...", fg=COLOR_PRIMARY
            )
            self.root.update_idletasks()

            try:
                results = yolo_model.predict(source=item["img_bgr"], conf=0.10, iou=0.30, verbose=False)
            except Exception as e:
                messagebox.showerror("YOLO ผิดพลาด", f"ภาพที่ {img_idx + 1}: {e}")
                continue

            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int) if len(results) else np.empty((0, 4), dtype=int)
            if len(boxes) == 0:
                item["processed"] = True
                item["cell_results"] = []
                item["display_boxed"] = item["img_bgr"].copy()
                item["display_heatmap"] = item["img_bgr"].copy()
                
                safe_name = f"image_{img_idx + 1}"
                image_dir = os.path.join(run_dir, f"img{img_idx + 1:02d}_{safe_name}")
                os.makedirs(image_dir, exist_ok=True)
                imwrite_unicode(os.path.join(image_dir, "full_slide_boxed.png"), item["img_bgr"])
                imwrite_unicode(os.path.join(image_dir, "full_slide_heatmap.png"), item["img_bgr"])
                continue

            h_img, w_img = item["img_bgr"].shape[:2]
            safe_name = f"image_{img_idx + 1}"
            image_dir = os.path.join(run_dir, f"img{img_idx + 1:02d}_{safe_name}")
            os.makedirs(image_dir, exist_ok=True)

            cell_results = []
            display_img = item["img_bgr"].copy()
            
            thickness = 1
            font_scale = 0.30
            infected_n, uninfected_n = 0, 0

            for i, (x1, y1, x2, y2) in enumerate(boxes):
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_img, x2), min(h_img, y2)
                if x2 - x1 < 4 or y2 - y1 < 4:
                    continue
                crop = item["img_bgr"][y1:y2, x1:x2].copy()

                try:
                    class_idx, confidence, model, img_input = classify_cell(crop, mode)
                except Exception as e:
                    messagebox.showerror("ข้อผิดพลาด", str(e))
                    continue

                label_key = "parasitized" if class_idx == 0 else "uninfected"
                if class_idx == 0:
                    infected_n += 1
                    box_color = (0, 0, 255)
                else:
                    uninfected_n += 1
                    box_color = (0, 255, 0)

                cv2.rectangle(display_img, (x1, y1), (x2, y2), box_color, thickness)
                cv2.putText(display_img, f"#{i+1}", (x1, max(10, y1 - 3)),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, box_color, 1, cv2.LINE_AA)

                filename = f"cell_{i + 1:03d}_{label_key}_{confidence:.1f}pct.png"
                filepath = os.path.join(image_dir, filename)
                imwrite_unicode(filepath, crop)

                cell_results.append({
                    "idx": i + 1, "bbox": (x1, y1, x2, y2), "crop": crop,
                    "class_idx": class_idx, "confidence": confidence,
                    "filename": filename, "filepath": filepath, "mode": mode,
                })

            item["cell_results"] = cell_results
            item["display_boxed"] = display_img
            item["infected"] = infected_n
            item["uninfected"] = uninfected_n
            item["processed"] = True

            infected_cells = [c for c in cell_results if c["class_idx"] == 0]
            if infected_cells:
                images_with_infection += 1
                self.big_verdict.config(
                    text=f"⏳ กำลังสร้างแผนที่ความร้อนภาพที่ {img_idx + 1}/{total_images}...", fg=COLOR_PRIMARY
                )
                self.root.update_idletasks()
                item["display_heatmap"] = build_infection_heatmap_image(item["img_bgr"], infected_cells)
            else:
                item["display_heatmap"] = display_img.copy()

            imwrite_unicode(os.path.join(image_dir, "full_slide_boxed.png"), item["display_boxed"])
            imwrite_unicode(os.path.join(image_dir, "full_slide_heatmap.png"), item["display_heatmap"])

            agg_total += infected_n + uninfected_n
            agg_infected += infected_n
            agg_uninfected += uninfected_n

        elapsed = time.time() - start_time

        self.stat_images.config(text=str(total_images))
        self.stat_total.config(text=str(agg_total))
        self.stat_infected.config(text=str(agg_infected))
        self.stat_uninfected.config(text=str(agg_uninfected))
        rate = (agg_infected / agg_total * 100) if agg_total else 0

        if agg_infected > 0:
            self.big_verdict.config(
                text=f"🦠 พบเซลล์ติดเชื้อรวม {agg_infected} เซลล์ ใน {images_with_infection}/{total_images} ภาพ",
                fg=COLOR_RED
            )
        else:
            self.big_verdict.config(text="✅ ไม่พบเซลล์ติดเชื้อในภาพทั้งหมด", fg=COLOR_GREEN)
        self.rate_label.config(text=f"อัตราติดเชื้อรวม {rate:.1f}% | ⏱️ {elapsed:.2f} วินาที\n(บันทึกที่ {run_dir})")

        self._refresh_image_selector()
        if self.smear_selected_idx is None and self.smear_items:
            self.smear_selected_idx = 0
        self._sync_selector_to_index()
        self._show_selected_smear_image()

        messagebox.showinfo("เสร็จสิ้น", f"ประมวลผลครบ {total_images} ภาพแล้ว\n⏱️ เวลาที่ใช้: {elapsed:.2f} วินาที")

    def _render_gallery(self):
        self.gallery.clear()
        item = self._current_item()
        if item is None:
            return
        filt = self.gallery_filter_var.get()
        items = item["cell_results"]
        if filt == "infected":
            items = [c for c in items if c["class_idx"] == 0]
        elif filt == "uninfected":
            items = [c for c in items if c["class_idx"] == 1]
        if filt == "all":
            items = sorted(items, key=lambda c: c["class_idx"])

        cols = 8
        for pos, cell in enumerate(items):
            r, c = divmod(pos, cols)
            cell_frame = tk.Frame(self.gallery.inner, bg=COLOR_CARD, padx=4, pady=4,
                                  highlightbackground=COLOR_RED if cell["class_idx"] == 0 else COLOR_GREEN,
                                  highlightthickness=2)
            cell_frame.grid(row=r, column=c, padx=5, pady=5)

            thumb = cv2_to_tk_thumb(cell["crop"], 88)
            btn = tk.Button(cell_frame, image=thumb, relief="flat", bd=0,
                            command=lambda c=cell: self.show_cell_detail(c))
            btn.image = thumb
            btn.pack()

            tag = "ติดเชื้อ" if cell["class_idx"] == 0 else "ปกติ"
            tk.Label(cell_frame, text=f"#{cell['idx']} {tag}",
                     font=(FONT_TH, 8, "bold"), bg=COLOR_CARD,
                     fg=COLOR_RED if cell["class_idx"] == 0 else COLOR_GREEN).pack()
            tk.Label(cell_frame, text=cell["filename"], font=(FONT_TH, 7), bg=COLOR_CARD,
                     fg="#90a4ae", wraplength=90).pack()

        if not items:
            msg = "ไม่มีเซลล์ในหมวดนี้" if item["processed"] else "ยังไม่ได้ประมวลผลภาพนี้ — กดปุ่มตัดเซลล์ + จำแนกทั้งหมด"
            tk.Label(self.gallery.inner, text=msg, font=(FONT_TH, 10),
                     bg=COLOR_BG, fg="#90a4ae").pack(padx=20, pady=20)

    def show_cell_detail(self, cell):
        win = tk.Toplevel(self.root)
        win.title(f"รายละเอียดเซลล์ #{cell['idx']}  —  {cell['filename']}")
        win.configure(bg=COLOR_BG)
        win.geometry("760x480")
        win.resizable(False, False)

        is_infected = cell["class_idx"] == 0
        color = COLOR_RED if is_infected else COLOR_GREEN
        banner = tk.Frame(win, bg=color, pady=12)
        banner.pack(fill="x")
        tk.Label(banner, text=CLASS_TH[cell["class_idx"]], font=(FONT_TH, 17, "bold"),
                 bg=color, fg="white").pack()
        tk.Label(banner, text=f"ความมั่นใจ {cell['confidence']:.2f}%    |    ไฟล์: {cell['filename']}",
                 font=(FONT_TH, 10), bg=color, fg="white").pack()

        body = tk.Frame(win, bg=COLOR_BG)
        body.pack(fill="both", expand=True, padx=10, pady=10)

        left = tk.Frame(body, bg=COLOR_CARD, padx=8, pady=8)
        left.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        ltop_sub = tk.Frame(left, bg=COLOR_CARD)
        ltop_sub.pack(fill="x", pady=(0, 6))
        tk.Label(ltop_sub, text="ภาพเซลล์ที่ตัดออกมา", font=(FONT_TH, 10, "bold"), bg=COLOR_CARD).pack(side="left")
        
        canvas_crop = ZoomPanCanvas(left, bg="#e8ecef", width=320, height=280)
        canvas_crop.pack(fill="both", expand=True)
        canvas_crop.set_image(cell["crop"])

        right = tk.Frame(body, bg=COLOR_CARD, padx=8, pady=8)
        right.pack(side="left", fill="both", expand=True, padx=(5, 0))
        
        rtop_sub = tk.Frame(right, bg=COLOR_CARD)
        rtop_sub.pack(fill="x", pady=(0, 6))
        tk.Label(rtop_sub, text="Grad-CAM Heatmap", font=(FONT_TH, 10, "bold"), bg=COLOR_CARD).pack(side="left")
        tk.Button(rtop_sub, text="💾 บันทึกรูปภาพ", command=lambda: self.save_canvas_image(canvas_cam),
                  font=(FONT_TH, 8, "bold"), bg=COLOR_GREEN, fg="white", relief="flat", padx=6, pady=2).pack(side="right")
        
        canvas_cam = ZoomPanCanvas(right, bg="#e8ecef", width=320, height=280)
        canvas_cam.pack(fill="both", expand=True)

        win.update_idletasks()

        try:
            model = color_model if cell["mode"] == "color" else bw_model
            img_input = prepare_input(cell["crop"], cell["mode"])
            heatmap = make_gradcam_heatmap(img_input, model)
            cam_img = overlay_heatmap(cell["crop"], heatmap, size=(300, 300))
            canvas_cam.set_image(cam_img)
        except Exception as e:
            print(f"⚠️ Grad-CAM ล้มเหลว: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = MalariaApp(root)
    root.mainloop()