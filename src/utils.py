import numpy as np
from PIL import Image, ImageOps

def preprocess_image(image):
    # Convert to grayscale
    img = image.convert("L")

    # Invert colors (white digit on black)
    img = ImageOps.invert(img)

    # Crop digit
    bbox = img.getbbox()
    if bbox is None:
        return None
    img = img.crop(bbox)

    # Resize to 28x28
    img = ImageOps.fit(img, (28, 28), centering=(0.5, 0.5))

    img = np.array(img).astype("float32")
    img = img / 255.0
    img = img.reshape(1, -1)

    return img
