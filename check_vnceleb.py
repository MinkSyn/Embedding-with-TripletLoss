import os

from tqdm import tqdm
from PIL import Image, ImageOps

from tool import get_tfms

ROOT_PATH = '/kaggle/input/vn-celeb/VN_celeb/not_mask'

transforms = get_tfms()

for celeb in os.listdir(ROOT_PATH):
    print(celeb)
    celeb_path = os.path.join(ROOT_PATH, celeb)
    
    for img_name in tqdm(os.listdir(celeb_path)):
        path = os.path.join(celeb_path, img_name)
        try:
            with open(path, 'rb') as f:
                img = Image.open(f).convert('RGB')
                img_gray = ImageOps.grayscale(img)
                
            img_trans = transforms(img_gray)
        except:
            print(f"Error [{path}]")