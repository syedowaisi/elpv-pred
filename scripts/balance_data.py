import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

train_dir = "dataset/split_data/train"

datagen = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Get class folders
classes = os.listdir(train_dir)
class_counts = {}

# Count images
for cls in classes:
    path = os.path.join(train_dir, cls)
    class_counts[cls] = len(os.listdir(path))

print("Before balancing:", class_counts)

max_count = max(class_counts.values())

# Augment minority classes
for cls in classes:
    class_path = os.path.join(train_dir, cls)
    images = os.listdir(class_path)
    current_count = len(images)

    if current_count < max_count:
        needed = max_count - current_count
        print(f"Augmenting {cls}: need {needed} images")

        i = 0
        while needed > 0:
            img_name = images[i % len(images)]
            img_path = os.path.join(class_path, img_name)

            img = load_img(img_path, target_size=(224,224))
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)

            aug_iter = datagen.flow(
                x,
                batch_size=1,
                save_to_dir=class_path,
                save_prefix="aug",
                save_format="jpg"
            )

            next(aug_iter)
            needed -= 1
            i += 1

print("Balancing complete!")
