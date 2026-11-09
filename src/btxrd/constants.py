"""Project-wide constants."""

CLASS_NAMES = [
    "normal",
    "osteochondroma",
    "multiple_osteochondromas",
    "simple_bone_cyst",
    "giant_cell_tumor",
    "osteofibroma",
    "synovial_osteochondroma",
    "osteosarcoma",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
VALID_SPLITS = {"train", "val", "test"}
IMAGE_FEATURE_DIM = 1664
EXPECTED_KB_PROPOSITIONS = 68
