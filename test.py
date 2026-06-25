import torch

print("PyTorch Version:", torch.__version__)
print("Is CUDA available?", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU Device Name:", torch.cuda.get_device_name(0))
    print("Current Device ID:", torch.cuda.current_device())
else:
    print("PyTorch is running on the CPU. Check your installation steps.")
