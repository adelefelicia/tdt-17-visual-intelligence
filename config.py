
# DATA CONFIGURATION
DATA_ROOT = "/cluster/projects/vc/courses/TDT17/mic/ODELIA2025/data"

SEQUENCES = ["Post_1", "Post_2", "Pre", "Sub_1", "T2"]

NUM_CLASSES = 3
CLASS_NAMES = ["normal", "benign", "malignant"]

IMAGE_SHAPE = (256, 256, 32)  # (H, W, D)
SPACING = (0.7, 0.7, 3.0)  # mm (H, W, D)

# TRAINING CONFIGURATION

# Data loading
BATCH_SIZE = 4
NUM_WORKERS = 4
CACHE_DATA = False
