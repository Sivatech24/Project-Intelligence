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

# -----------------------------
# SEGMENT G: MULTI-STEP & META RULES
# -----------------------------

def rule_composition_crop_and_rotate(grid):
    """Executes cropping, then rotates."""
    cropped = cropping_remove_background(grid)
    return rotate_90_clockwise(cropped)

def analogy_solver(grid):
    """
    ABSTRACT RULE: "A is to B as C is to D"
    Cannot be solved via deterministic single-grid function.
    Requires the engine to analyze the relationships between train_input and train_output.
    """
    # Placeholder: Returns input safely to avoid crashing the engine.
    return grid.copy()

def relational_reasoning(grid):
    """Placeholder for abstract relational reasoning."""
    return grid.copy() # Placeholder

def goal_directed_behavior(grid):
    """
    ABSTRACT RULE: e.g., "Move blue dot to red square."
    Requires pathfinding (A* algorithm) and semantic labeling.
    """
    # Placeholder: Returns input safely.
    return grid.copy()

# -----------------------------
# RULE COMPILATION LIST
# -----------------------------

RULES = [
    # Base Rules (from the provided base code)
    ("add_one", add_one),
    ("subtract_one", subtract_one),
    ("flip_horizontal", flip_horizontal),
    ("flip_vertical", flip_vertical),
    ("rotate_90_base", rotate_90), # Renamed to avoid clash with rotate_90_clockwise
    ("rotate_180", rotate_180),
    ("identity", identity),

    # Color Transformation Rules
    ("color_swap_most_least", color_mapping_swap_most_least),
    ("conditional_binarize", conditional_coloring_binarize),
    ("color_mapping_generic", color_mapping_generic),

    # Spatial Transformation Rules
    ("rotate_90", rotate_90_clockwise),
    ("reflect_horiz", reflection_horizontal),
    ("reflect_vert", reflection_vertical),
    ("shift_right", translation_shift_right),
    ("shift_left", translation_shift_left),
    ("shift_up", translation_shift_up),
    ("shift_down", translation_shift_down),
    ("crop_background", cropping_remove_background),
    ("resizing_generic", resizing_generic),

    # Object-Based Rules
    ("isolate_largest_object", object_detection_isolate_largest),
    ("object_detection_generic", object_detection_generic),
    ("count_objects", object_counting_to_pixel),
    ("object_movement", object_movement),
    ("object_duplication", object_duplication),
    ("object_scaling", object_scaling),
    ("align_push_left", object_alignment_push_left),
    ("align_push_right", object_alignment_push_right),
    ("align_push_top", object_alignment_push_top),
    ("align_push_bottom", object_alignment_push_bottom),

    # Pattern & Symmetry Rules
    ("symmetry_mirror", symmetry_creation_mirror_right),
    ("pattern_repetition", pattern_repetition),
    ("tile_2x2", grid_tiling_2x2),

    # Logical / Conditional & Math Rules
    ("conditional_rule_example", conditional_rule_example),
    ("count_total_pixels", counting_total_pixels),
    ("arithmetic_x2", arithmetic_multiply_by_two),
    ("arithmetic_generic", arithmetic_generic),

    # Geometry, Shape & Noise Rules
    ("shape_detection", shape_detection),
    ("shape_completion", shape_completion),
    ("shortest_path", shortest_path),
    ("grid_partitioning", grid_partitioning),
    ("projection", projection),
    ("extract_edges", border_rules_extract_edges),
    ("remove_single_pixels", remove_noise_single_pixels),
    ("keep_important_objects", keep_important_objects),

    # Multi-Step & Meta Rules
    ("crop_and_rotate", rule_composition_crop_and_rotate),
    ("analogy", analogy_solver),
    ("relational_reasoning", relational_reasoning),
    ("goal_directed", goal_directed_behavior)
]

# -----------------------------
# RULE MATCHING ENGINE
# -----------------------------

def find_rule(train_input, train_output):
    first_matching_rule = None
    print("\n--- Evaluating All Rules for Train Example ---")
    for name, rule in RULES:
        start_time = time.time()
        try:
            predicted = rule(train_input)
            end_time = time.time()
            execution_time = (end_time - start_time) * 1000 # in milliseconds
            if np.array_equal(predicted, train_output):
                print(f"Rule: {name} (MATCHED - {execution_time:.2f} ms)")
                if first_matching_rule is None:
                    first_matching_rule = rule
            else:
                print(f"Rule: {name} (NO MATCH - {execution_time:.2f} ms)")
        except Exception as e:
            end_time = time.time()
            execution_time = (end_time - start_time) * 1000 # in milliseconds
            print(f"Rule: {name} (ERROR - {execution_time:.2f} ms): {e}")
            continue
    print("--- End Rule Evaluation ---\n")
    return first_matching_rule

# -----------------------------
# SOLVER FUNCTION
# -----------------------------

def solve_arc(train_input, train_output, test_input):
    rule = find_rule(train_input, train_output)

    print(f"solve_arc received rule: {rule.__name__ if rule else 'None'}") # Added for debugging

    if rule is None:
        print("No matching rule found")
        return None

    result = rule(test_input)
    return result

