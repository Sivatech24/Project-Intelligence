import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# -----------------------------
# 1. MODEL (Vision-based)
# -----------------------------

class ARCNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 10, 1)  # 10 colors
        )

    def forward(self, x):
        return self.model(x)

# -----------------------------
# 2. DATA PREP
# -----------------------------

def to_tensor(grid):
    return torch.tensor(grid).long()

def prepare_data(train_pairs):
    X, Y = [], []

    for inp, out in train_pairs:
        X.append(to_tensor(inp))
        Y.append(to_tensor(out))

    X = torch.stack(X).unsqueeze(1).float()  # (B,1,H,W)
    Y = torch.stack(Y)                       # (B,H,W)

    return X, Y

# -----------------------------
# 3. TEST-TIME TRAINING
# -----------------------------

def train_on_task(model, X, Y, epochs=200):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        optimizer.zero_grad()

        output = model(X)  # (B,10,H,W)
        loss = criterion(output, Y)

        loss.backward()
        optimizer.step()

    return model

# -----------------------------
# 4. PREDICTION
# -----------------------------

def predict(model, test_input):
    x = torch.tensor(test_input).unsqueeze(0).unsqueeze(0).float()
    output = model(x)

    pred = torch.argmax(output, dim=1)
    return pred.squeeze().detach().numpy()

# -----------------------------
# 5. SOLVER
# -----------------------------

def solve_arc(train_pairs, test_input):
    model = ARCNet()

    X, Y = prepare_data(train_pairs)

    # Learn rule per task
    model = train_on_task(model, X, Y)

    result = predict(model, test_input)

    return result

# -----------------------------
# 6. TEST CASE
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
print("\nPredicted Output:\n", result)
