import numpy as np
import imageio as img
import matplotlib.pyplot as plt

image = img.imread("https://images.unsplash.com/photo-1500964757637-c85e8a162699?fm=jpg&q=60&w=3000&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxleHBsb3JlLWZlZWR8M3x8fGVufDB8fHx8fA%3D%3D")

grayscale_image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

plt.figure(figsize=(8, 8))
plt.imshow(grayscale_image, cmap='gray')
plt.title("Gambar Grayscale")
plt.axis("off")
plt.show()

histogram, bin_edges = np.histogram(grayscale_image, bins=256, range=(0, 255))

plt.figure(figsize=(10, 6))
plt.bar(bin_edges[:-1], histogram, width=1, color='black', alpha=0.7)
plt.xlim(0, 255)
plt.title("Histogram Grayscale")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

total_pixels_per_intensity = dict(zip(bin_edges[:-1], histogram))

for intensity, total in total_pixels_per_intensity.items():
    print(f"Intensitas: {int(intensity)} | Total Piksel: {total}")
