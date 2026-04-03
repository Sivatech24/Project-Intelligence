import numpy as np

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
    return grid.copy()

# List of rules
RULES = [
    ("add_one", add_one),
    ("subtract_one", subtract_one),
    ("flip_horizontal", flip_horizontal),
    ("flip_vertical", flip_vertical),
    ("rotate_90", rotate_90),
    ("rotate_180", rotate_180),
    ("identity", identity),
]

# -----------------------------
# 2. RULE MATCHING ENGINE
# -----------------------------

def find_rule(train_input, train_output):
    for name, rule in RULES:
        try:
            predicted = rule(train_input)
            if np.array_equal(predicted, train_output):
                print(f"Rule Found: {name}")
                return rule
        except:
            continue
    return None

# -----------------------------
# 3. SOLVER FUNCTION
# -----------------------------

def solve_arc(train_input, train_output, test_input):
    rule = find_rule(train_input, train_output)

    if rule is None:
        print("No matching rule found")
        return None

    result = rule(test_input)
    return result

# -----------------------------
# 4. TEST CASE
# -----------------------------

# Training example
train_input = np.array([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0]
])

train_output = np.array([
    [1, 2, 1],
    [2, 1, 2],
    [1, 2, 1]
])

# Test input
test_input = np.array([
    [2, 0, 1],
    [1, 2, 0],
    [0, 1, 2]
])

# -----------------------------
# 5. RUN SOLVER
# -----------------------------

result = solve_arc(train_input, train_output, test_input)

print("\nTest Input:\n", test_input)
print("\nPredicted Output:\n", result)
