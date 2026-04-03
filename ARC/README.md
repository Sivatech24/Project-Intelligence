# PyTorch_ARC_Model.py:

Epoch 1, Loss: 3.6467
Epoch 2, Loss: 2.5218
Epoch 3, Loss: 1.6148
Epoch 4, Loss: 0.9705
Epoch 5, Loss: 0.6850
Epoch 6, Loss: 0.7930
Epoch 7, Loss: 1.0513
Epoch 8, Loss: 1.1470
Epoch 9, Loss: 1.0423
Epoch 10, Loss: 0.8446

Input:
 tensor([[0., 2., 1.],
        [1., 0., 2.],
        [2., 1., 0.]])

Predicted Output:
 tensor([[1., 2., 2.],
        [3., 3., 3.],
        [2., 2., 1.]])

# ARC_Solver.py:

Rule Found: add_one

Test Input:
 [[2 0 1]
 [1 2 0]
 [0 1 2]]

Predicted Output:
 [[3 1 2]
 [2 3 1]
 [1 2 3]]

# ARC Solver.py:

Found rule: color_map

Input:
 [[2 0]
 [1 2]]

Output:
 [[2 1]
 [2 2]]

# Hybrid ARC Solver.py:

Input:
 [[2 0]
 [1 2]]

Predicted Output:
 [[1 1]
 [2 1]]
