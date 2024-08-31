import torch
import torch.nn as nn

# Load the model
model_path = "Web/Models/model_84_acc_10_frames_final_data.pt"
model = torch.load(model_path)
model.eval()  # Set the model to evaluation mode

# Define a hook function to capture feature maps
def hook_fn(module, input, output):
    # Store the output in a global variable or a list
    feature_maps.append(output)

# Register hooks on desired layers
feature_maps = []  # This will store the feature maps
hook_handles = []

def register_hooks(model):
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):  # Adjust based on your model
            handle = layer.register_forward_hook(hook_fn)
            hook_handles.append(handle)

register_hooks(model)

# Example input tensor (adjust size as needed for your model)
input_tensor = torch.randn(1, 3, 224, 224)  # Example size for an image model
with torch.no_grad():
    model(input_tensor)  # Perform a forward pass

# feature_maps now contains the output from each hooked layer
for idx, fmap in enumerate(feature_maps):
    print(f"Feature map {idx}: shape = {fmap.shape}")

# Optionally, remove hooks if no longer needed
for handle in hook_handles:
    handle.remove()



