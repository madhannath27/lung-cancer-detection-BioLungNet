import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.densenet import preprocess_input


model_path = r"D:\BioLungNet\Program\Model\BioLungNet_Model.h5"
model = load_model(model_path)
print(f"✅ Model loaded from: {model_path}")

class_labels = ['Benign', 'Malignant', 'Normal']


def select_image():
    root = tk.Tk()
    root.withdraw()  # Hide empty window

    file_path = filedialog.askopenfilename(
        title="Select a CT Scan Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    return file_path


# ---------------- Prediction Function ----------------
def predict_image(model, image_path, class_labels):
    img = load_img(image_path, target_size=(224, 224))
    img_rgb = img.convert('RGB')
    img_array = img_to_array(img_rgb)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    preds = model.predict(img_array, verbose=0)
    pred_idx = np.argmax(preds, axis=1)[0]
    pred_class = class_labels[pred_idx]
    confidence = preds[0][pred_idx]

    return img_rgb, pred_class, confidence


# ------------------- MAIN -------------------
image_path = select_image()

if image_path:
    print(f"\n📌 Selected Image: {image_path}")

    img_rgb, pred_class, confidence = predict_image(model, image_path, class_labels)

    # Show image + result
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title(f"{os.path.basename(image_path)}\nPredicted: {pred_class} ({confidence*100:.2f}%)")
    plt.show()

    # Save result to CSV
    results = [{
        "Filename": os.path.basename(image_path),
        "Predicted Class": pred_class,
        "Confidence (%)": round(confidence * 100, 2)
    }]

    csv_path = r"D:\NEW\AUGMENTED DATASET\single_image_prediction.csv"
    pd.DataFrame(results).to_csv(csv_path, index=False)

    print(f"\n✅ Prediction saved to CSV: {csv_path}")

else:
    print("❌ No image selected.")
