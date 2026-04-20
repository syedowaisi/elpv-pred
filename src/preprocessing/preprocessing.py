import os
import cv2

# Input dataset
input_root = r"C:\Users\HP\Desktop\minor_project\dataset\split_data"

# Output dataset
output_root = r"C:\Users\HP\Desktop\minor_project\src\preprocessing\processed_data"

# Target image size
IMG_SIZE = 64   # ✅ changed

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

            # Resize
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            # Convert grayscale → 3 channel
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            # Normalize
            img = img / 255.0

            # Convert back for saving
            img = (img * 255).astype("uint8")

            save_path = os.path.join(output_folder, img_name)

            cv2.imwrite(save_path, img)

print("✅ Preprocessing completed successfully (64x64)")