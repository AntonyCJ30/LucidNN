"""
Run this once to generate test_model.pth — a simple 2->4->4->1 network.
Then upload test_model.pth in the LucidNN app to test the import feature.
"""
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
)

# Save state_dict (the format LucidNN imports)
torch.save(model.state_dict(), "test_model.pth")
print("Saved test_model.pth  — topology: 2 → 4 → 4 → 1")
print("Layers in state_dict:")
for k, v in model.state_dict().items():
    print(f"  {k:30s}  shape={tuple(v.shape)}")
