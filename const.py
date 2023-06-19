from enum import Enum


IMG_SIZE = (128, 128)

STATS = {
    'imagenet': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}


class QualityID(Enum):
    blur = 0
    occluded = 1
    overexposed = 2
    sharp = 3
    shadow = 4
    outofdate = 5
    lowquality = 6
    overclosed = 7


class CardID(Enum):
    CitizenCardV1_back = 0
    CitizenCardV1_front = 1
    CitizenCardV2_back = 2
    CitizenCardV2_front = 3
    IdentificationCard_back = 4
    IdentificationCard_front = 5
