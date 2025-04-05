import cv2
import os
import numpy as np

def load_images_from_folder(folder_path, img_size=128):
    images = []
    labels = []
    classes = os.listdir(folder_path)

    for label, class_name in enumerate(classes):
        class_folder = os.path.join(folder_path, class_name)
        for filename in os.listdir(class_folder):
            img_path = os.path.join(class_folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (img_size, img_size))
                img = img / 255.0
                images.append(img)
                labels.append(label)

    images = np.array(images).reshape(-1, img_size, img_size, 1)
    return images, np.array(labels)