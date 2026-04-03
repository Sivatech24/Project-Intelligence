import torch
import torch.nn as nn
import torch.optim as optim

# ---- 1. Create dummy ARC dataset ----
# Each grid is 3x3

def generate_data(num_samples=1000):
    X = []
    Y = []
    for _ in range(num_samples):
        grid = torch.randint(0, 3, (3, 3))  # values 0–2
        target = grid + 1                  # transformation rule
        X.append(grid)
        Y.append(target)
    return torch.stack(X), torch.stack(Y)

X, Y = generate_data()

# Normalize to float
X = X.float()
Y = Y.float()

# ---- 2. Simple Model (CNN like vision approach) ----
class ARCModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1)
        )

    def forward(self, x):
        return self.net(x)

model = ARCModel()

# ---- 3. Prepare data ----
# Add channel dimension → (batch, channel, height, width)
X = X.unsqueeze(1)
Y = Y.unsqueeze(1)

# ---- 4. Training ----
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(10):
    optimizer.zero_grad()
    
    output = model(X)
    loss = criterion(output, Y)
    
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# ---- 5. Test ----
test_input = torch.tensor([
    [0, 2, 1],
    [1, 0, 2],
    [2, 1, 0]
]).float().unsqueeze(0).unsqueeze(0)

prediction = model(test_input)

print("\nInput:\n", test_input.squeeze())
print("\nPredicted Output:\n", prediction.detach().round().squeeze())
