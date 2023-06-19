import os
from PIL import Image, ImageOps

from tool import get_tfms

ROOT_PATH = ''

transforms = get_tfms()

for celeb in os.listdir(ROOT_PATH):
    celeb_path = os.path.join(ROOT_PATH, celeb)
    
    for img_name in os.listdir(celeb_path):
        path = os.path.join(celeb_path, img_name)
        try:
            with open(path, 'rb') as f:
                img = Image.open(f).convert('RGB')
                img_gray = ImageOps.grayscale(img)
                
            img_trans = transforms(img_gray)
        except:
            print(f"Error [{path}]")