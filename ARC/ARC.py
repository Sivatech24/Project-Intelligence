import numpy as np
from scipy.ndimage import label, find_objects, binary_erosion
import time

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

def identity(grid):
    """Returns the grid unchanged."""
    return grid.copy()

# -----------------------------
# SEGMENT A: COLOR TRANSFORMATION RULES
# -----------------------------

def color_mapping_swap_most_least(grid):
    """Swaps the most frequent color with the least frequent color (ignoring 0/background)."""
    if np.all(grid == 0): return grid
    unique, counts = np.unique(grid[grid != 0], return_counts=True)
    if len(unique) < 2: return grid.copy()

    most_frequent = unique[np.argmax(counts)]
    least_frequent = unique[np.argmin(counts)]

    new_grid = grid.copy()
    new_grid[grid == most_frequent] = least_frequent
    new_grid[grid == least_frequent] = most_frequent
    return new_grid

def conditional_coloring_binarize(grid):
    """Turns all non-zero pixels to 1 (useful for silhouetting)."""
    return np.where(grid > 0, 1, 0)

def color_mapping_generic(grid):
    """Placeholder for a generic color mapping rule."""
    return grid.copy() # Placeholder
