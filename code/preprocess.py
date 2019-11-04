from PIL import Image
import numpy as np 

def load_image(image_path):
    img = Image.open(image_path)
    img = np.asarray(img)/255
    
    return img