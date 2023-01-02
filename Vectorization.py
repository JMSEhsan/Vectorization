# Reference: Tiago Rodrigues Antao, Fast Python for Data Science, Manning, 2022

import numpy as np
from PIL import Image

image = Image.open("mySun.jpg")
width, height = image.size
image.show()
image_ms = np.array(image)
print(image_ms.shape, image_ms.dtype)

def get_grayscale_color(row):
    mean = np.mean(row)
    return int(mean)

vec_get_grayscale_color = np.vectorize(get_grayscale_color, otypes = [np.uint8], signature ="(n)->()")
grayscale_ms = vec_get_grayscale_color(image_ms)
print(np.array(grayscale_ms).max(), np.array(grayscale_ms).dtype, np.array(grayscale_ms).shape)
Image.fromarray(grayscale_ms).save("grayscale_mySun.png")
image_gs = Image.open("grayscale_mySun.png")
image_gs.show()