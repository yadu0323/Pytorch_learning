import torch

# 1. Define the device target (defaults to CPU if GPU isn't available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Convert Data Tensors to GPU
# Old CPU way: x = torch.tensor([1.0, 2.0, 3.0])
x = torch.tensor([1.0, 2.0, 3.0]).to(device)
y = torch.tensor([4.0, 5.0, 6.0]).to(device)

# 3. Operations now automatically happen on the GPU
z = x + y
print("Result Tensor:", z)
print("Tensor Device:", z.device) # Will output 'cuda:0'

# 4. Convert Neural Network Models to GPU
# class MyModel(torch.nn.Module): ...
# model = MyModel()
# model.to(device) <--- Pushes all model weights to the GPU
