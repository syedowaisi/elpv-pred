import pandas as pd
import os
import shutil

csv_path = "dataset/labels.csv"
image_folder = "dataset/images"
output_base = "dataset/sorted_data"

# create folders
class_folders = ["functional", "moderate", "mild", "severe"]
for c in class_folders:
    os.makedirs(os.path.join(output_base, c), exist_ok=True)

# read space-separated csv
df = pd.read_csv(csv_path, header=None, sep=r"\s+")

for _, row in df.iterrows():
    image_path = row[0]       # images/cell0001.png
    label = float(row[1])    # e.g. 0.3333333 / 0.6666667

    filename = os.path.basename(image_path)

    # ----- RANGE BASED CLASSIFICATION -----
    if label < 0.17:
        class_name = "functional"
    elif label < 0.5:
        class_name = "moderate"
    elif label < 0.83:
        class_name = "mild"
    else:
        class_name = "severe"

    src = os.path.join(image_folder, filename)
    dst = os.path.join(output_base, class_name, filename)

    if os.path.exists(src):
        shutil.copy(src, dst)
    else:
        print("Missing:", filename)

print("Images sorted successfully!")
