import tensorflow as tf
import cv2
import numpy as np
import glob, random

model = tf.keras.models.load_model("malaria_final_model_bw.h5")

def test_folder(folder_path, true_label, n=30):
    files = glob.glob(folder_path)
    random.seed(42)
    sample = random.sample(files, min(n, len(files)))
    correct = 0
    for path in sample:
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (128, 128))
        inp = np.expand_dims(np.expand_dims(resized / 255.0, axis=-1), axis=0)
        pred = model.predict(inp, verbose=0)
        pred_class = int(np.argmax(pred[0]))
        status = "ถูก" if pred_class == true_label else "ผิด"
        print(f"{path.split(chr(92))[-1]:35s} pred={pred[0]}  -> {status}")
        if pred_class == true_label:
            correct += 1
    acc = correct / len(sample) * 100
    print(f"\n=== {folder_path} ===")
    print(f"ทายถูก {correct}/{len(sample)} = {acc:.1f}%\n")

# true_label=0 คือ "ติดเชื้อ" ตามโค้ดแอปหลัก (class_idx==0)
test_folder(r"D:\Malaria_Python_App\Train_Dataset_BW\Parasitized\*.png", true_label=0, n=30)
test_folder(r"D:\Malaria_Python_App\Train_Dataset_BW\Uninfected\*.png", true_label=1, n=30)