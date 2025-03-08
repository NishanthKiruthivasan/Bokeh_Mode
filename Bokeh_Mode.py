import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

# Load MiDaS depth estimation model
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()

# Preprocessing function
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def estimate_depth(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize image
        transforms.ToTensor(),          # Convert image to tensor
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize
    ])

    img_resized = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img_resized = Image.fromarray(img_resized)  # Convert NumPy array to PIL Image
    img_resized = transform(img_resized).unsqueeze(0)  # Apply transforms

    with torch.no_grad():
        prediction = midas(img_resized)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()

    return prediction.cpu().numpy()

def apply_variable_bokeh(image, depth_map):
    """Applies different levels of blur based on depth."""
    # Define depth thresholds for different blur intensities
    near_threshold = np.percentile(depth_map, 40)  # Foreground
    mid_threshold = np.percentile(depth_map, 60)   # Midground
    far_threshold = np.percentile(depth_map, 80)   # Background

    # Create masks for different depth ranges
    near_mask = (depth_map > near_threshold).astype(np.uint8) * 255
    mid_mask = ((depth_map > mid_threshold) & (depth_map <= near_threshold)).astype(np.uint8) * 255
    far_mask = (depth_map <= mid_threshold).astype(np.uint8) * 255

    # Apply different blur intensities
    blurred_mid = cv2.GaussianBlur(image, (15, 15), 10)  # Moderate blur
    blurred_far = cv2.GaussianBlur(image, (25, 25), 20)  # Stronger blur

    # Combine images based on depth
    result = np.where(near_mask[..., None] == 255, image, blurred_mid)
    result = np.where(far_mask[..., None] == 255, blurred_far, result)

    return result

# Load input image
image_path = "input.jpg"  # Replace with your image path
image = cv2.imread(image_path)

# Estimate depth
depth_map = estimate_depth(image)

# Apply variable bokeh effect
bokeh_image = apply_variable_bokeh(image, depth_map)

# Save and show the output
cv2.imwrite("output.jpg", bokeh_image)
cv2.waitKey(0)
cv2.destroyAllWindows()