import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# Create a torch.Tensor object with the given data
V_data = [1., 2., 3.]
V = torch.Tensor(V_data)
print(V)

# Matrix
M_data = [[1., 2., 3], [4., 5., 6]]
M = torch.Tensor(M_data)
print(M)

# create 3D tensor of size
T_data = [[[1., 2.], [3., 4.]],
          [[5., 6.], [7., 8.]]]
T = torch.Tensor(T_data)
print(T)

# Index into V and get a scalar
print(V[0])

# Index into M and get a vector
print(M[0])

# Index into T and get a matrix
print(T[0])

x = torch.randn((3, 4, 5))
print(x)

x = torch.Tensor([1., 2., 3.])
y = torch.Tensor([4., 5., 6.])
z = x + y
print(z)

# Cat first axis
x_1 = torch.randn(2, 5)
y_1 = torch.randn(3, 5)
z_1 = torch.cat([x_1, y_1])
print(z_1)

x_2 = torch.randn(2, 3)
y_2 = torch.randn(2, 5)
z_2 = torch.cat([x_2, y_2], 1)
print(z_2)

x = torch.randn(2, 3, 4)
print(x)
print(x.view(2, 12))
print(x.view(2, -1))

# variable wrap tensor object
x = autograd.Variable(torch.Tensor([1., 2., 3]), requires_grad=True)
print(x)

y = autograd.Variable(torch.Tensor([4., 5., 6.]), requires_grad=True)
z = x + y
print(z.data)

print(z.creator)

s = z.sum()
print(s)
print(s.creator)

s.backward()
print(x.grad)

x = torch.randn((2, 2))
y = torch.randn((2, 2))
z = x + y

var_x = autograd.Variable(x)
var_y = autograd.Variable(y)
var_z = var_x + var_y
print(var_z.creator)
