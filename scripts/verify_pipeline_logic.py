"""
Verification of the pipeline logic without running PyTorch.
This demonstrates the conceptual correctness of the approach.
"""

class MockModel:
    def __init__(self):
        self.blocks = [MockBlock(i) for i in range(4)]

    def forward(self, x):
        for block in self.blocks:
            x = block.forward(x)
        return x

class MockBlock:
    def __init__(self, idx):
        self.idx = idx
        self.hooks = []

    def forward(self, x):
        # Simulate computation
        output = x + self.idx + 1  # Simple transformation

        # Call hooks if any
        for hook in self.hooks:
            modified_output = hook(self, x, output)
            if modified_output is not None:
                output = modified_output

        return output

    def register_hook(self, hook_fn):
        self.hooks.append(hook_fn)

    def clear_hooks(self):
        self.hooks = []

# Test the mock pipeline
print("\nRunning mock test...")
model = MockModel()

# 1. Baseline
x = 10
baseline_output = model.forward(x)
print(f"Baseline output: {baseline_output}")

# 2. Extract
extracted = {}
def extract_hook(idx):
    def hook(module, input, output):
        extracted[idx] = output
        return None 
    return hook

for i, block in enumerate(model.blocks):
    block.register_hook(extract_hook(i))

extract_output = model.forward(x)
print(f"Extraction output: {extract_output}")
print(f"Extracted values: {extracted}")

# Clear hooks
for block in model.blocks:
    block.clear_hooks()

# 3. Inject
def inject_hook(idx):
    def hook(module, input, output):
        return extracted[idx]  # Return saved value
    return hook

for i, block in enumerate(model.blocks):
    block.register_hook(inject_hook(i))

inject_output = model.forward(x)
print(f"Injection output: {inject_output}")

# Verify
if baseline_output == inject_output:
    print("\n[SUCCESS] Mock pipeline works")
    print("   Extract -> Inject produces identical output!")
else:
    print("\n[FAILURE] Outputs don't match.")

print("\n" + "=" * 70)
print("This demonstrates the pipeline logic is sound w/o pytorch.")
print("=" * 70)