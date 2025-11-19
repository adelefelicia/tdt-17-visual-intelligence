
# DATA CONFIGURATION
DATA_ROOT = "/cluster/projects/vc/courses/TDT17/mic/ODELIA2025/data"

NUM_SEQUENCES = 5
SEQUENCES = ["Post_1", "Post_2", "Pre", "Sub_1", "T2"]

NUM_CLASSES = 3
CLASS_NAMES = ["normal", "benign", "malignant"]

IMAGE_SHAPE = (256, 256, 32)  # (H, W, D)
SPACING = (0.7, 0.7, 3.0)  # mm (H, W, D)

# TRAINING CONFIGURATION

# Training hyperparameters
NUM_EPOCHS = 100
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-5
USE_CLASS_WEIGHTS = True

# Learning rate scheduler
WARMUP_EPOCHS = 5
WARMUP_END_LR = 0.0001
MIN_LR = 1e-6

# Early stopping
EARLY_STOPPING_PATIENCE = 20

# Data loading
BATCH_SIZE = 4
NUM_WORKERS = 4
CACHE_DATA = False

# Misc
RANDOM_SEED = 42
