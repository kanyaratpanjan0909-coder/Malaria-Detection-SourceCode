import os
import cv2
import glob
import numpy as np
from ultralytics import YOLO

INPUT_FOLDER = 'slide_input'
CONF_THRESHOLD = 0.20   # 🔥 ลดลงมาที่ 20% เพื่อบังคับให้มันใจดีขึ้น รูปไหนจางๆ ก็ต้องยอมตัดออกมา
IOU_THRESHOLD = 0.35    # ตั้งค่า NMS ป้องกันกล่องซ้อนทับกันเกินไป
LINE_WIDTH = 1

model_path = 'best.pt' 
model = YOLO(model_path)

valid_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.PNG']
raw_image_files = []
for ext in valid_extensions:
    raw_image_files.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))

image_files = []
seen_paths = set()
for img_path in raw_image_files:
    norm_path = os.path.normpath(img_path).lower()
    if norm_path not in seen_paths:
        seen_paths.add(norm_path)
        image_files.append(img_path)

if not image_files:
    print(f"❌ ไม่พบไฟล์รูปภาพใน '{INPUT_FOLDER}'")
    exit()

output_main_folder = 'cropped_results'
os.makedirs(output_main_folder, exist_ok=True)
total_cells = 0

for img_idx, img_path in enumerate(image_files, 1):
    img_name = os.path.basename(img_path)
    name_without_ext = os.path.splitext(img_name)[0]
    img_output_folder = os.path.join(output_main_folder, name_without_ext)
    os.makedirs(img_output_folder, exist_ok=True)
    
    image = cv2.imread(img_path)
    if image is None: continue
    
    # รัน YOLO รอบแรกด้วยเกณฑ์ 0.20
    results = model.predict(img_path, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)
    
    # เช็คว่าเจอเซลล์ไหม ถ้าภาพไหนโล่งเตียน (0 boxes) ให้ลองลดเกณฑ์ลงเหลือ 0.10 เป็นกรณีพิเศษทันที!
    total_boxes = sum(len(r.boxes) for r in results)
    if total_boxes == 0:
        print(f"⚠️ รูป {img_name} ไม่พบเซลล์ที่เกณฑ์ 0.20 กำลังลองสแกนรอบสองด้วยความไวสูง (Conf=0.10)...")
        results = model.predict(img_path, conf=0.10, iou=IOU_THRESHOLD)
        
    img_cell_count = 0
    for result in results:
        annotated_image = image.copy()
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf_val = float(box.conf[0])
            class_id = int(box.cls[0])
            
            try:
                class_name = model.names[class_id]
            except:
                class_name = "cell"

            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (255, 0, 0), LINE_WIDTH)
            text = f"{conf_val:.2f}"
            cv2.putText(annotated_image, text, (x1, max(y1 - 2, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)

            x1_c, y1_c = max(0, x1), max(0, y1)
            x2_c, y2_c = min(image.shape[1], x2), min(image.shape[0], y2)
            cropped_cell = image[y1_c:y2_c, x1_c:x2_c]
            w_c = x2_c - x1_c
            h_c = y2_c - y1_c

            if cropped_cell.size > 0:
                img_cell_count += 1
                total_cells += 1
                
                save_path = os.path.join(img_output_folder, f"crop_{class_name}_{img_cell_count}_X{x1_c}_Y{y1_c}_W{w_c}_H{h_c}.jpg")
                cv2.imwrite(save_path, cropped_cell)

        overview_path = os.path.join(img_output_folder, f"overview_{img_name}")
        cv2.imwrite(overview_path, annotated_image)

print(f"✨ ประมวลผลเสร็จสิ้น! ตัดเซลล์รวมทั้งหมด {total_cells} เซลล์")