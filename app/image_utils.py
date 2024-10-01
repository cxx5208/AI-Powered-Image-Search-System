from PIL import Image
import os

def save_image(file, save_path):
    """
    Save uploaded image to disk.
    """
    img = Image.open(file)
    img.save(save_path)
