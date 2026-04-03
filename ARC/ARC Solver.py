import numpy as np
from collections import Counter

# -----------------------------
# 1. BASIC RULES
# -----------------------------

def flip_horizontal(grid):
    return np.fliplr(grid)

def flip_vertical(grid):
    return np.flipud(grid)

def rotate_90(grid):
    return np.rot90(grid)

def identity(grid):
    return grid.copy()

# -----------------------------
# 2. COLOR MAPPING (IMPORTANT)
# -----------------------------

def get_color_mapping(input_grid, output_grid):
    mapping = {}
    for i in range(input_grid.shape[0]):
        for j in range(input_grid.shape[1]):
            inp = input_grid[i, j]
            out = output_grid[i, j]
            if inp not in mapping:
                mapping[inp] = out
            elif mapping[inp] != out:
                return None  # inconsistent mapping
    return mapping

def apply_color_mapping(grid, mapping):
    new_grid = grid.copy()
    for k, v in mapping.items():
        new_grid[grid == k] = v
    return new_grid

# -----------------------------
# 3. OBJECT DETECTION (BFS)
# -----------------------------

def find_objects(grid):
    visited = np.zeros_like(grid, dtype=bool)
    objects = []

    rows, cols = grid.shape

    def bfs(r, c):
        color = grid[r, c]
        queue = [(r, c)]
        obj = []
        visited[r, c] = True

        while queue:
            x, y = queue.pop(0)
            obj.append((x, y))

            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols:
                    if not visited[nx, ny] and grid[nx, ny] == color:
                        visited[nx, ny] = True
                        queue.append((nx, ny))
        return obj, color

    for i in range(rows):
        for j in range(cols):
            if not visited[i, j]:
                obj, color = bfs(i, j)
                objects.append((obj, color))

    return objects

# -----------------------------
# 4. RULE ENGINE
# -----------------------------

def try_basic_rules(inp, out):
    rules = [
        ("identity", identity),
        ("flip_h", flip_horizontal),
        ("flip_v", flip_vertical),
        ("rotate_90", rotate_90),
    ]

    for name, rule in rules:
        if np.array_equal(rule(inp), out):
            return (name, rule)
    return None

def try_color_rule(inp, out):
    mapping = get_color_mapping(inp, out)
    if mapping:
        return ("color_map", mapping)
    return None

# -----------------------------
# 5. COMBINED RULES
# -----------------------------

def apply_rule(rule, grid):
    name, func = rule

    if name == "color_map":
        return apply_color_mapping(grid, func)
    else:
        return func(grid)

def try_combined_rules(inp, out):
    basic_rules = [
        identity,
        flip_horizontal,
        flip_vertical,
        rotate_90
    ]

    for r1 in basic_rules:
        temp = r1(inp)

        mapping = get_color_mapping(temp, out)
        if mapping:
            return ("combo", (r1, mapping))

    return None

# -----------------------------
# 6. MAIN SOLVER
# -----------------------------

def find_rule(train_pairs):
    # Try all training examples
    for inp, out in train_pairs:

        # 1. Basic rule
        r = try_basic_rules(inp, out)
        if r:
            return r

        # 2. Color rule
        r = try_color_rule(inp, out)
        if r:
            return r

        # 3. Combined rule
        r = try_combined_rules(inp, out)
        if r:
            return r

    return None

def apply_final_rule(rule, grid):
    name = rule[0]

    if name == "combo":
        r1, mapping = rule[1]
        return apply_color_mapping(r1(grid), mapping)

    return apply_rule(rule, grid)

def solve_arc(train_pairs, test_input):
    rule = find_rule(train_pairs)

    if rule is None:
        print("No rule found")
        return None

    print(f"Found rule: {rule[0]}")
    return apply_final_rule(rule, test_input)

# -----------------------------
# 7. TEST CASE
# -----------------------------

train_pairs = [
    (
        np.array([[0,1],[1,0]]),
        np.array([[1,2],[2,1]])
    )
]

test_input = np.array([
    [2,0],
    [1,2]
])

result = solve_arc(train_pairs, test_input)

print("\nInput:\n", test_input)
print("\nOutput:\n", result)
