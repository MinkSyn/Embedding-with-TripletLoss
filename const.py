from enum import Enum


IMG_SIZE = (300, 450)

STATS = {
    'image_net': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}

class FolderID(Enum):
    folder1 = 0
    folder2 = 1
    
class QualityID(Enum):
    blur = 1
    normal = 0
    