import torch
import torch.nn as nn

# 1. Define a simple neural network class
class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(20, 5)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

# 2. Create an instance of the model
model = ToyModel()

# 3. Prepare a random input tensor (batch size 1, 10 features)
x = torch.randn(1, 10)

# 4. Run a forward pass and print the original output
original_output = model(x)
print("Original model output:", original_output)

# 5. Define a patching hook for linear1
#    This will replace the output of linear1 with a custom tensor

def patch_hook(module, input, output):
    print("\n[Patch Hook] Original linear1 output:", output)
    # Example: zero out the activation
    patched = torch.zeros_like(output)
    print("[Patch Hook] Patched linear1 output:", patched)
    return patched  # This will replace the output of linear1

# 6. Register the patching hook (PyTorch 2.0+)
patch_handle = model.linear1.register_forward_hook(patch_hook)

# 7. Run a forward pass with patching and print the new output
patched_output = model(x)
print("\nModel output after patching linear1:", patched_output)

# 8. Remove the hook
patch_handle.remove()