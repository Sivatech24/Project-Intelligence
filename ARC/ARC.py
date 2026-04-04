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

# -----------------------------
# SEGMENT C: OBJECT-BASED RULES
# -----------------------------

def object_detection_isolate_largest(grid):
    """Returns a grid containing ONLY the largest object, removing the rest."""
    labeled_grid, num_features = get_objects(grid)
    if num_features == 0: return grid

    # Count sizes of each labeled object
    sizes = np.bincount(labeled_grid.ravel())
    sizes[0] = 0 # Ignore background
    largest_label = sizes.argmax()

    return np.where(labeled_grid == largest_label, grid, 0)

def object_detection_generic(grid):
    """Placeholder for a generic object detection rule."""
    return grid.copy() # Placeholder

def object_counting_to_pixel(grid):
    """Counts objects and outputs a 1x1 grid with that number."""
    _, num_features = get_objects(grid)
    return np.array([[num_features]])

def object_movement(grid):
    """Placeholder for a rule that moves objects."""
    return grid.copy() # Placeholder

def object_duplication(grid):
    """Placeholder for a rule that duplicates objects."""
    return grid.copy() # Placeholder

def object_scaling(grid):
    """Placeholder for a rule that scales objects."""
    return grid.copy() # Placeholder

def object_alignment_push_left(grid):
    """Pushes all pixels as far left as possible in their respective rows."""
    new_grid = np.zeros_like(grid)
    for r in range(grid.shape[0]):
        row = grid[r]
        non_zeros = row[row != 0]
        new_grid[r, :len(non_zeros)] = non_zeros
    return new_grid

def object_alignment_push_right(grid):
    """Placeholder: Pushes all pixels as far right as possible."""
    return grid.copy() # Placeholder

def object_alignment_push_top(grid):
    """Placeholder: Pushes all pixels as far top as possible."""
    return grid.copy() # Placeholder

def object_alignment_push_bottom(grid):
    """Placeholder: Pushes all pixels as far bottom as possible."""
    return grid.copy() # Placeholder

# -----------------------------
# SEGMENT D: PATTERN & SYMMETRY RULES
# -----------------------------

def symmetry_creation_mirror_right(grid):
    """Takes the grid and mirrors it to the right, doubling its width."""
    mirrored = np.fliplr(grid)
    return np.hstack((grid, mirrored))

def pattern_repetition(grid):
    """Placeholder for a rule that repeats patterns."""
    return grid.copy() # Placeholder

def grid_tiling_2x2(grid):
    """Tiles the current grid into a 2x2 larger grid."""
    return np.tile(grid, (2, 2))

# -----------------------------
# SEGMENT E: LOGICAL / CONDITIONAL & MATH
# -----------------------------

def conditional_rule_example(grid):
    """Placeholder for an if-else logic rule example."""
    return grid.copy() # Placeholder

def counting_total_pixels(grid):
    """Returns 1x1 grid of total non-zero pixels."""
    return np.array([[np.count_nonzero(grid)]])

def arithmetic_multiply_by_two(grid):
    """Multiplies all pixel values (colors) by 2."""
    # Modulo 10 because ARC colors are usually 0-9
    return (grid * 2) % 10

def arithmetic_generic(grid):
    """Placeholder for a generic arithmetic operation."""
    return grid.copy() # Placeholder

# -----------------------------
# SEGMENT F: GEOMETRY, SHAPE & NOISE
# -----------------------------

def shape_detection(grid):
    """Placeholder for a rule that detects specific shapes."""
    return grid.copy() # Placeholder

def shape_completion(grid):
    """Placeholder for a rule that completes shapes."""
    return grid.copy() # Placeholder

def shortest_path(grid):
    """Placeholder for a shortest path or path drawing rule."""
    return grid.copy() # Placeholder

def grid_partitioning(grid):
    """Placeholder for a rule that partitions the grid."""
    return grid.copy() # Placeholder

def projection(grid):
    """Placeholder for a rule that projects grid elements."""
    return grid.copy() # Placeholder

def border_rules_extract_edges(grid):
    """Keeps only the outer edges of shapes, hollowing them out."""
    if np.all(grid == 0): return grid

    mask = grid != 0
    eroded = binary_erosion(mask)
    edges = mask ^ eroded # XOR to find difference
    return np.where(edges, grid, 0)

def remove_noise_single_pixels(grid):
    """Removes stray single pixels (objects of size 1)."""
    labeled_grid, num_features = get_objects(grid)
    sizes = np.bincount(labeled_grid.ravel())
    noise_labels = np.where(sizes == 1)[0]

    new_grid = grid.copy()
    for label_idx in noise_labels:
        if label_idx == 0: continue
        new_grid[labeled_grid == label_idx] = 0
    return new_grid

def keep_important_objects(grid):
    """Placeholder for a rule that keeps only important objects (e.g., based on size, color)."""
    return grid.copy() # Placeholder

