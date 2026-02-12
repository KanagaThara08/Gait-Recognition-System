import os
import cv2
import numpy as np
from skimage.feature import hog
from tqdm import tqdm

def load_dataset(dataset_path):
    X = []
    y = []

    print("[INFO] Scanning for PNG files...")

    all_image_paths = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(".png"):
                all_image_paths.append(os.path.join(root, file))

    print(f"[INFO] Found {len(all_image_paths)} PNG files.")
    print("[INFO] Loading dataset with progress bar...\n")

    for image_path in tqdm(all_image_paths, desc="Processing Images", unit="img"):
        try:
            subject_id = os.path.basename(os.path.dirname(os.path.dirname(image_path)))
        except IndexError:
            continue

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        image = cv2.resize(image, (128, 128))
        features = hog(
            image,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            visualize=False
        )

        X.append(features)
        y.append(subject_id)

    print(f"\n[INFO] Finished loading. Total Samples: {len(X)} | Total Subjects: {len(set(y))}")
    return np.array(X), np.array(y)
