import os
import cv2
import shutil

# ------------------------------
# Input dataset
# ------------------------------
input_root = r"C:\Users\HP\Desktop\minor_project\dataset\split_data"

# ------------------------------
# Output dataset
# ------------------------------
output_root = r"C:\Users\HP\Desktop\minor_project\src\preprocessing\processed_data"

# ------------------------------
# 🔥 DELETE OLD DATA (IMPORTANT)
# ------------------------------
if os.path.exists(output_root):
    shutil.rmtree(output_root)

# ------------------------------
# Target image size
# ------------------------------
IMG_SIZE = 224   # ✅ updated

splits = ["train", "val", "test"]
classes = ["functional", "moderate", "mild", "severe"]

for split in splits:
    for cls in classes:

        input_folder = os.path.join(input_root, split, cls)
        output_folder = os.path.join(output_root, split, cls)

        os.makedirs(output_folder, exist_ok=True)

        for img_name in os.listdir(input_folder):

            img_path = os.path.join(input_folder, img_name)

            # Read as grayscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            # Resize to 224×224
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            # Convert grayscale → 3-channel (RGB)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            # Save directly (no need to normalize here)
            save_path = os.path.join(output_folder, img_name)
            cv2.imwrite(save_path, img)

print("✅ Preprocessing completed successfully (224x224)")