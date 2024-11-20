BOX_TYPES = ["all", "small", "medium", "large"]
BOX_TYPE_TO_SIZE = {
    "all": [0, 1e10],
    "small": [0, 32**2],
    "medium": [32**2, 96**2],
    "large": [96**2, 1e10],
}
