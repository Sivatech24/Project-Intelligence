import numpy as np
from scipy.ndimage import label, find_objects, binary_erosion

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

