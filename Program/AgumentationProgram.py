
import os
import cv2
from tqdm import tqdm
import albumentations as A

# -------- Paths --------
input_paths = {
    "Normal": r"F:\@MADHAN\Project\DATASET\ORIGINAL DATASET\Normal cases",
    "Benign": r"F:\@MADHAN\Project\DATASET\ORIGINAL DATASET\Benign cases",
    "Malignant": r"F:\@MADHAN\Project\DATASET\ORIGINAL DATASET\Malignant cases"
}

output_paths = {
    "Normal": r"F:\@MADHAN\Project\lung_cancer_ctscan\AUGMENTED DATASET\Normal",
    "Benign": r"F:\@MADHAN\Project\lung_cancer_ctscan\AUGMENTED DATASET\Benign",
    "Malignant": r"F:\@MADHAN\Project\lung_cancer_ctscan\AUGMENTED DATASET\Malignant"
}

for cls in output_paths.keys():
    os.makedirs(output_paths[cls], exist_ok=True)

target_count = 3500

# -------- Updated Augmentation Pipeline --------
augmenter = A.Compose([
    A.CLAHE(clip_limit=2.0, p=0.4),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HorizontalFlip(p=0.5),
    # Replace ShiftScaleRotate with Affine
    A.Affine(scale=(0.9, 1.1), translate_percent=(0.0, 0.08), rotate=(-20, 20), p=0.7),
    A.GridDistortion(p=0.3),
    A.GaussNoise(var_limit=(5, 25), mean=0, p=0.3),   # keep var_limit (still valid in 2.x)
    A.ElasticTransform(alpha=50, sigma=10, p=0.3),
    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
    A.CoarseDropout(
        max_holes=8,
        hole_height_range=(8, 16),
        hole_width_range=(8, 16),
        fill_value=0,
        p=0.4
    ),
    A.Resize(224, 224),
    A.Normalize(mean=(0.5,), std=(0.5,))
])

# -------- Augmentation Loop --------
for cls in input_paths.keys():
    images = [f for f in os.listdir(input_paths[cls]) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    input_folder = input_paths[cls]
    output_folder = output_paths[cls]

    n_original = len(images)
    if n_original == 0:
        print(f"⚠️ No images found for {cls}")
        continue

    aug_per_image = max(1, (target_count // n_original))
    print(f"🧩 Augmenting {cls}: {n_original} originals → aiming for ~{target_count} images ({aug_per_image} per image)")

    for img_name in tqdm(images):
        img_path = os.path.join(input_folder, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        for i in range(aug_per_image):
            augmented = augmenter(image=img)
            aug_img = augmented['image']

            save_name = f"{os.path.splitext(img_name)[0]}_aug{i}.jpg"
            save_path = os.path.join(output_folder, save_name)
            cv2.imwrite(save_path, (aug_img * 255).astype('uint8'))

    print(f"✅ {cls} class done → total {len(os.listdir(output_folder))} images saved.")

print("\n🎯 Advanced augmentation complete for all classes!")
