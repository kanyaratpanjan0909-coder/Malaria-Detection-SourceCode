import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import pandas as pd
import os

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 15

DIR_COLOR = "Train_Dataset_Color"
DIR_BW = "Train_Dataset_BW"

if not os.path.exists(DIR_COLOR) or not os.path.exists(DIR_BW):
    print("⚠️ ไม่พบโฟลเดอร์ Train_Dataset_Color หรือ Train_Dataset_BW กรุณาตรวจสอบชื่อโฟลเดอร์อีกครั้งครับคิม!")
else:
    print("🚀 เริ่มต้นกระบวนการเทรนโมเดลสำหรับระบบ Grad-CAM และบันทึกผลการทดลอง...")

    # ==========================================
    # 1. เทรนโมเดลภาพสี (Color Model)
    # ==========================================
    print("\n--- [1/2] กำลังเทรนโมเดลภาพสี (Color Model) ---")
    
    train_datagen_color = ImageDataGenerator(
        rescale=1.0/255.0, validation_split=0.2,
        rotation_range=20, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True
    )

    train_gen_color = train_datagen_color.flow_from_directory(
        DIR_COLOR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', color_mode='rgb', subset='training'
    )
    val_gen_color = train_datagen_color.flow_from_directory(
        DIR_COLOR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', color_mode='rgb', subset='validation'
    )

    color_model = models.Sequential([
        layers.Input(shape=(128, 128, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'), # เลเยอร์ Conv สุดท้ายสำหรับดึงค่า Grad-CAM
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(2, activation='softmax')
    ])

    color_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history_color = color_model.fit(train_gen_color, validation_data=val_gen_color, epochs=EPOCHS)
    color_model.save("malaria_final_model.h5")
    print("✅ บันทึกโมเดลภาพสีสำเร็จ: malaria_final_model.h5")

    # ==========================================
    # 2. เทรนโมเดลภาพขาวดำ (Grayscale Model)
    # ==========================================
    print("\n--- [2/2] กำลังเทรนโมเดลภาพขาวดำ (Grayscale Model) ---")
    
    train_datagen_bw = ImageDataGenerator(
        rescale=1.0/255.0, validation_split=0.2,
        rotation_range=20, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True
    )

    train_gen_bw = train_datagen_bw.flow_from_directory(
        DIR_BW, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', color_mode='grayscale', subset='training'
    )
    val_gen_bw = train_datagen_bw.flow_from_directory(
        DIR_BW, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', color_mode='grayscale', subset='validation'
    )

    bw_model = models.Sequential([
        layers.Input(shape=(128, 128, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'), # เลเยอร์ Conv สุดท้ายสำหรับดึงค่า Grad-CAM
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(2, activation='softmax')
    ])

    bw_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history_bw = bw_model.fit(train_gen_bw, validation_data=val_gen_bw, epochs=EPOCHS)
    bw_model.save("malaria_final_model_bw.h5")
    print("✅ บันทึกโมเดลภาพขาวดำสำเร็จ: malaria_final_model_bw.h5")

    # ==========================================
    # 3. แสดงผลตารางสรุปผลรายรอบ (Epoch) สำหรับใส่เล่มบทที่ 4
    # ==========================================
    print("\n" + "="*85)
    print("📊 ตารางสรุปผลการเทรนโมเดลภาพสี (Color Model) - สำหรับนำไปใส่ในเล่มบทที่ 4")
    print("="*85)
    df_color = pd.DataFrame({
        "Epoch": list(range(1, EPOCHS + 1)),
        "Loss": [f"{val:.4f}" for val in history_color.history['loss']],
        "Accuracy": [f"{val:.4f}" for val in history_color.history['accuracy']],
        "Val_Loss": [f"{val:.4f}" for val in history_color.history['val_loss']],
        "Val_Accuracy": [f"{val:.4f}" for val in history_color.history['val_accuracy']]
    })
    print(df_color.to_string(index=False))
    print("="*85)

    print("\n" + "="*85)
    print("📊 ตารางสรุปผลการเทรนโมเดลภาพขาวดำ (Grayscale Model) - สำหรับนำไปใส่ในเล่มบทที่ 4")
    print("="*85)
    df_bw = pd.DataFrame({
        "Epoch": list(range(1, EPOCHS + 1)),
        "Loss": [f"{val:.4f}" for val in history_bw.history['loss']],
        "Accuracy": [f"{val:.4f}" for val in history_bw.history['accuracy']],
        "Val_Loss": [f"{val:.4f}" for val in history_bw.history['val_loss']],
        "Val_Accuracy": [f"{val:.4f}" for val in history_bw.history['val_accuracy']]
    })
    print(df_bw.to_string(index=False))
    print("="*85)

    # ==========================================
    # 4. สร้างและบันทึกรูปกราฟผลการทดลอง
    # ==========================================
    # กราฟภาพสี
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history_color.history['accuracy'], label='Train Accuracy')
    plt.plot(history_color.history['val_accuracy'], label='Val Accuracy')
    plt.title('Color Model - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history_color.history['loss'], label='Train Loss')
    plt.plot(history_color.history['val_loss'], label='Val Loss')
    plt.title('Color Model - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('color_model_performance.png')

    # กราฟภาพขาวดำ
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history_bw.history['accuracy'], label='Train Accuracy')
    plt.plot(history_bw.history['val_accuracy'], label='Val Accuracy')
    plt.title('Grayscale Model - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history_bw.history['loss'], label='Train Loss')
    plt.plot(history_bw.history['val_loss'], label='Val Loss')
    plt.title('Grayscale Model - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('bw_model_performance.png')

    print("\n📈 บันทึกรูปกราฟเรียบร้อย: 'color_model_performance.png' และ 'bw_model_performance.png'")
    print("🎉 กระบวนการทั้งหมดเสร็จสมบูรณ์ พร้อมใช้งานร่วมกับระบบ Grad-CAM แล้วครับคิม!")