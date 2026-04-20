import os
import random
import shutil

source_dir = "dataset/sorted_data"
target_dir = "dataset/split_data"

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

classes = os.listdir(source_dir)

for cls in classes:
    images = os.listdir(os.path.join(source_dir, cls))
    random.shuffle(images)

    train_count = int(len(images) * train_ratio)
    val_count = int(len(images) * val_ratio)

    train_imgs = images[:train_count]
    val_imgs = images[train_count:train_count+val_count]
    test_imgs = images[train_count+val_count:]

    for folder, img_list in zip(["train","val","test"], [train_imgs,val_imgs,test_imgs]):
        dst_folder = os.path.join(target_dir, folder, cls)
        os.makedirs(dst_folder, exist_ok=True)

        for img in img_list:
            src = os.path.join(source_dir, cls, img)
            dst = os.path.join(dst_folder, img)
            shutil.copy(src, dst)

print("Dataset split completed!")