import cv2
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import ace_tools as tools
import pandas as pd


# Create synthetic biochip images
def generate_biochip_image(
    grid_size=(8, 8),
    img_size=256,
    spot_radius=8,
    intensity_range=(50, 255),
    noise_level=10,
):
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    label = []

    step = img_size // (grid_size[0] + 1)
    center_points = []

    for i in range(1, grid_size[0] + 1):
        for j in range(1, grid_size[1] + 1):
            cx, cy = step * i, step * j
            intensity = np.random.randint(intensity_range[0], intensity_range[1])
            label.append(1 if intensity > 150 else 0)
            cv2.circle(img, (cx, cy), spot_radius, (intensity,), -1)
            center_points.append((cx, cy))

    # Add Gaussian noise
    noise = np.random.normal(0, noise_level, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)

    return img, np.array(label).reshape(grid_size)


# Generate dataset of 100 images
os.makedirs("/mnt/data/biochip_images", exist_ok=True)
labels = []
for i in tqdm(range(100)):
    img, label = generate_biochip_image()
    labels.append(label)
    cv2.imwrite(f"/mnt/data/biochip_images/biochip_{i}.png", img)

# Display a sample
plt.imshow(img, cmap="gray")
plt.title("Synthetic Biochip Example")
plt.axis("off")
plt.show()

# Save labels
labels = np.array(labels)
np.save("/mnt/data/biochip_images/labels.npy", labels)

tools.display_dataframe_to_user(
    name="Biochip Labels", dataframe=pd.DataFrame(labels.reshape(100, -1))
)
