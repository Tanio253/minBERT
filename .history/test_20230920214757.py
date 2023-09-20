import torch
import torch.nn as nn

# Create tensors
tensor1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
tensor2 = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])

# Dropout layer
dropout_prob = 0.1
dropout1 = nn.Dropout(p=dropout_prob)
dropout2 = nn.Dropout(p = dropout_prob)
# Apply dropout to the original tensors
tensor1_dropout = dropout1(tensor1)
tensor2_dropout = dropout2(tensor2)

# Apply the same dropout to the original tensors again
tensor1_dropout_again = dropout1(tensor1)
tensor2_dropout_again = dropout1(tensor2)

print("Tensor 1 (Original):", tensor1)
print("Tensor 1 (After dropout):", tensor1_dropout)
print("Tensor 1 (After dropout again):", tensor1_dropout_again)
print()

print("Tensor 2 (Original):", tensor2)
print("Tensor 2 (After dropout):", tensor2_dropout)
print("Tensor 2 (After dropout again):", tensor2_dropout_again)
