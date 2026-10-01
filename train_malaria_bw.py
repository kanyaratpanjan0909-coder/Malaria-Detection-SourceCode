import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt
import os

# --- 1. ตั้งค่าพารามิเตอร์หลักสำหรับภาพขาวดำ ---
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 15
TRAIN_DIR = "Train_Dataset_BW"  # ⚠️ เปลี่ยนชื่อโฟลเดอร์ให้ตรงกับโฟลเดอร์ภาพขาวดำของคิม

print("--- เริ่มกระบวนการเตรียมข้อมูลและฝึกสอนโมเดลภาพขาวดำสำหรับบทที่ 4 ---")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    validation_split=0.2,
    subset="training",
    seed=123
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    validation_split=0.2,
    subset="validation",
    seed=123
)

class_names = train_dataset.class_names
print(f"🎯 คลาสเป้าหมายทั้งหมด: {class_names}")

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# --- 2. สร้างโครงสร้างโมเดล (MobileNetV2 สำหรับภาพขาวดำ) ---
# หมายเหตุ: ช่อง input_shape ยังคงเป็น 3 แชนเนล (RGB) ไว้ก่อนเพื่อให้เข้ากับโครงสร้าง MobileNetV2 ได้ทันทีโดยไม่ต้องแก้สถาปัตยกรรมซับซ้อนครับ
print("กำลังสร้างสถาปัตยกรรมโมเดล CNN สำหรับภาพขาวดำ...")
base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
base_model.trainable = True

model = models.Sequential([
    layers.Rescaling(1./127.5, offset=-1, input_shape=(128, 128, 3)),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --- 3. เริ่มต้นเทรนโมเดล ---
print(f"🚀 เริ่มต้นการเทรนโมเดลภาพขาวดำทั้งหมด {EPOCHS} รอบ (Epochs)...")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS
)

# บันทึกไฟล์โมเดลภาพขาวดำ
model.save("malaria_final_model_bw.h5")
print("✅ บันทึกไฟล์โมเดลภาพขาวดำเสร็จสิ้น: malaria_final_model_bw.h5")

# --- 4. พิมพ์ตารางสรุปผล ---
print("\n" + "="*50)
print("📊 ตารางสรุปผลการทดลอง (ภาพขาวดำ)")
print("="*50)
print(f"{'Epoch':<8} | {'Loss':<10} | {'Accuracy':<10} | {'Val_Loss':<10} | {'Val_Accuracy':<12}")
print("-" * 55)
for i in range(EPOCHS):
    print(f"{i+1:<8} | {history.history['loss'][i]:<10.4f} | {history.history['accuracy'][i]:<10.4f} | {history.history['val_loss'][i]:<10.4f} | {history.history['val_accuracy'][i]:<12.4f}")
print("="*50)

# --- 5. วาดและบันทึกกราฟ Learning Curve สำหรับภาพขาวดำ ---
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.title('Model Accuracy per Epoch (BW)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
plt.title('Model Loss per Epoch (BW)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
graph_filename = "learning_curve_result_bw.png"
plt.savefig(graph_filename, dpi=300)
print(f"📈 บันทึกกราฟผลการทดลองภาพขาวดำเรียบร้อยแล้วในชื่อ: '{graph_filename}'")