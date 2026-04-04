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

# -----------------------------
# SEGMENT B: SPATIAL TRANSFORMATION RULES
# -----------------------------

def rotate_90_clockwise(grid):
    return np.rot90(grid, k=-1) # k=-1 is clockwise

def reflection_horizontal(grid):
    return np.fliplr(grid)

def reflection_vertical(grid):
    return np.flipud(grid)

def translation_shift_right(grid):
    """Shifts all pixels one position to the right."""
    return np.roll(grid, shift=1, axis=1)

def translation_shift_left(grid):
    """Placeholder: Shifts all pixels one position to the left."""
    return np.roll(grid, shift=-1, axis=1) # Placeholder

def translation_shift_up(grid):
    """Placeholder: Shifts all pixels one position up."""
    return np.roll(grid, shift=-1, axis=0) # Placeholder

def translation_shift_down(grid):
    """Placeholder: Shifts all pixels one position down."""
    return np.roll(grid, shift=1, axis=0) # Placeholder

def cropping_remove_background(grid):
    """Crops the grid to the bounding box of all non-zero pixels."""
    if np.all(grid == 0): return grid
    rows = np.any(grid, axis=1)
    cols = np.any(grid, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return grid[rmin:rmax+1, cmin:cmax+1]

def resizing_generic(grid):
    """Placeholder for a generic resizing rule."""
    return grid.copy() # Placeholder
