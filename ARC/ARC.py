import numpy as np
from scipy.ndimage import label, find_objects, binary_erosion

# --- Helper Functions ---
def get_objects(grid):
    """Helper: Identifies isolated shapes (objects) in a grid."""
    # Assuming 0 is background. Label contiguous non-zero pixels.
    labeled_grid, num_features = label(grid != 0)
    return labeled_grid, num_features

# -----------------------------
# 1. RULE DEFINITIONS
# -----------------------------

def add_one(grid):
    return grid + 1

def subtract_one(grid):
    return grid - 1

def flip_horizontal(grid):
    return np.fliplr(grid)

def flip_vertical(grid):
    return np.flipud(grid)

def rotate_90(grid):
    return np.rot90(grid)

def rotate_180(grid):
    return np.rot90(grid, 2)

