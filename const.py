from enum import Enum


IMG_SIZE = (300, 450)

STATS = {
    'patchcore_v3.6': ((0.6074, 0.616, 0.603), (0.2317, 0.2292, 0.237)),
    'v3.6': ((0.6056, 0.6146, 0.6020), (0.2276, 0.2248, 0.2327)),
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
