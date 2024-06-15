import torch

# ========================================================== #
#                    Initializing Tensor                     #
# ========================================================== #
# print(torch.__version__)
device = "cuda" if torch.cuda.is_available() else "cpu"

my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32,
                         device=device, requires_grad=True)
print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.device)
print(my_tensor.shape)
print(my_tensor.requires_grad)

# Other common initialization methods
x = torch.empty(size= (3, 3))
x = torch.zeros((3, 3))
x = torch.rand((3, 3))
x = torch.ones((3, 3))
x = torch.eye(5, 5)
x = torch.arange(start=0, end=5, step=1)
x = torch.linspace(start=0.1, end=1, steps=10)
x = torch.empty(size=(1, 5)).normal_(mean=0, std=1)
x = torch.empty(size=(1, 5)).uniform_(0, 1)
x = torch.diag(torch.ones(3))
# print(x)

# How to initialize and convert tensors to other types (int, float, double)
tensor = torch.arange(4)
print(tensor.bool()) # boolean True/False
print(tensor.short()) # int16
print(tensor.long()) # int64 (Important)
print(tensor.half()) # float16
print(tensor.float()) # float32 (Important)
print(tensor.double()) # float64

# Array to Tensor conversion and vice-versa
import numpy as np
np_array = np.zeros((5, 5))
print("numpy array: ", np_array)
tensor = torch.from_numpy(np_array)
print("numpy to Tensor: ", tensor)
np_array_back = tensor.numpy()
print("numpy array back: ", np_array_back)

# ======================================================================== #
#                Tensor Math & Comparison Operations                       #
# ======================================================================== #

x = torch.tensor([1, 2, 3])
y = torch.tensor([9, 8, 7])

# Addition
z1 = torch.empty(3)
torch.add(x, y, out=z1)
print(z1)

z2 = torch.add(x, y)
print(z2)
z = x + y

# Subtraction
z = x - y

# Division
z = torch.true_divide(x, y)

# inplace operation
t = torch.zeros(3)
t.add_(x) # _ means inplace
# or
t += x

# Exponentiation
z = x.pow(2)
z = x ** 2

# Simple comparison
z = x > 0
print(z)
z = x < 0

# Matrix Multiplication
x1 = torch.rand((2, 5))
x2 = torch.rand((5, 3))
x3 = torch.mm(x1, x2) # 2x3
x3 = x1.mm(x2)

# matrix exponentiation
matrix_exp = torch.rand(5, 5)
print(matrix_exp.matrix_exp())

# element wise mult.
z = x * y
print(z)

# dot product
z = torch.dot(x, y)

# Batch Matrix Multiplication
batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand((batch, n, m))
tensor2 = torch.rand((batch, m, p))
out_bmm = torch.bmm(tensor1, tensor2) # (batch, n, p)
print(out_bmm)

# Example of Broadcasting
x1 = torch.rand((5,5))
x2 = torch.rand((1,5))

z = x1 - x2
z = x1 ** x2

# Other useful tensor operations
sum_x = torch.sum(x, dim=0)
values, indices = torch.max(x, dim=0) # x.max(dim=0)
values, indices = torch.min(x, dim=0)
abs_x = torch.abs(x)
z = torch.argmax(x, dim=0)
z = torch.argmin(x, dim=0)
mean_x = torch.mean(x.float(), dim=0)
z = torch.eq(x, y)
sorted_y, indices = torch.sort(y, dim=0, descending=False)

z = torch.clamp(x, min=0, max=10)

x = torch.tensor([1, 0, 1, 1, 1], dtype=torch.bool)
z = torch.any(x)
z = torch.all(x)
print(z)

# ================================================================= #
#                      Tensor Indexing                              #
# ================================================================= #

batch_size = 10
features = 25
x = torch.rand((batch_size, features))

print(x[0].shape) # x[0,:]
print(x[:, 0].shape)

print(x[2, 0:10])
x[0, 0] = 100

# Fancy indexing
x = torch.arange(10)
indices = [2, 5, 8]
print(x[indices])

x = torch.rand((3, 5))
rows = torch.tensor([1, 0])
cols = torch.tensor([4, 0])
print(x[rows, cols].shape)

# More advanced indexing
x = torch.arange(10)
print(x[(x < 2) | (x > 8)])
print(x[(x < 2) & (x > 8)])
print(x[x.remainder(2) == 0])

# useful operations
print(torch.where(x > 5, x, x*2))
print(torch.tensor([0,0,1,2,2,3,4]).unique())
print(x.ndimension()) # 5x5x5
print(x.numel())

# =========================================================== #
#                     Tensor Reshaping                        #
# =========================================================== #

import torch

# Creating a tensor x with values from 0 to 8
x = torch.arange(9)

# Reshaping x into a 3x3 tensor using view
x_3x3 = x.view(3, 3)

# Reshaping x into a 3x3 tensor using reshape
x_3x3 = x.reshape(3, 3)

# Transposing the 3x3 tensor (swapping rows and columns)
y = x_3x3.t()

# Creating two random tensors of shape (2, 5)
x1 = torch.rand((2, 5))
x2 = torch.rand((2, 5))

# Concatenating x1 and x2 along the first dimension (rows)
# The resulting shape is (4, 5) because we stack them vertically
print(torch.cat((x1, x2), dim=0).shape) # Output: torch.Size([4, 5])

# Concatenating x1 and x2 along the second dimension (columns)
# The resulting shape is (2, 10) because we stack them horizontally
print(torch.cat((x1, x2), dim=1).shape) # Output: torch.Size([2, 10])

# Flattening x1 into a 1D tensor
z = x1.view(-1) # The view(-1) flattens the tensor into a single dimension

# Creating a random tensor x with shape (batch, 2, 5)
batch = 64
x = torch.rand((batch, 2, 5))

# Reshaping x into a tensor with shape (batch, 10)
# The -1 argument infers the size of that dimension
z = x.view(batch, -1)
print(z.shape) # Output: torch.Size([64, 10])

# Permuting the dimensions of x to swap the second and third dimensions
# The resulting shape is (batch, 5, 2)
z = x.permute(0, 2, 1)
print(z.shape) # Output: torch.Size([64, 5, 2])

# Creating a tensor x with values from 0 to 9
x = torch.arange(10)

# Adding a dimension at the 0th position
# The resulting shape is (1, 10)
print(x.unsqueeze(0).shape) # Output: torch.Size([1, 10])

# Adding a dimension at the 1st position
# The resulting shape is (10, 1)
print(x.unsqueeze(1).shape) # Output: torch.Size([10, 1])

# Chaining unsqueeze to add dimensions
# First unsqueeze adds a dimension at 0th position, resulting in shape (1, 10)
# Second unsqueeze adds a dimension at 1st position, resulting in shape (1, 1, 10)
x = torch.arange(10).unsqueeze(0).unsqueeze(1) # Shape: (1, 1, 10)

# Removing the dimension at the 1st position
# The resulting shape is (1, 10)
z = x.squeeze(1)
print(z.shape) # Output: torch.Size([1, 10])