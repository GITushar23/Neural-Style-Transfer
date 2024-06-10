import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Function to load and preprocess images
def img_loader(img_path, size):
    image = Image.open(img_path).convert('RGB')
    loader = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = loader(image)[:3, :, :].unsqueeze(0)
    return image.to(device, torch.float)

# Function to display images
def imshow(tensor, title=None):
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone().detach()
    image = image.numpy().squeeze(0)
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

# VGG19 model definition
class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        self.chosen_features = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '21': 'conv4_2',  # Content representation
            '28': 'conv5_1'
        }
        self.model = models.vgg19(pretrained=True).features

    def forward(self, x):
        features = {}
        for name, layer in self.model._modules.items():
            x = layer(x)
            if name in self.chosen_features:
                features[self.chosen_features[name]] = x
        return features

# Content loss function
def content_loss(target_features, generated_features):
    return torch.mean((target_features - generated_features) ** 2)

# Gram matrix function
def gram_matrix(input_tensor):
    b, c, h, w = input_tensor.size()
    features = input_tensor.view(b * c, h * w)
    return torch.mm(features, features.t()) / (b * c * h * w)

# Style loss function
def style_loss(target_features, generated_features):
    target_gram = gram_matrix(target_features)
    generated_gram = gram_matrix(generated_features)
    return torch.mean((target_gram - generated_gram) ** 2)

# Load content and style images
size = 356
content_image = img_loader('./content.png', size)
style_image = img_loader('./style.png', size)
# Initialize combination image (generated image)
target_tensor = content_image.clone().requires_grad_(True)
# target_tensor = torch.randn_like(content_image).to(device).requires_grad_(True)

# Define optimizer
optimizer = optim.Adam([target_tensor], lr=0.1)

# Define VGG19 model
model = VGG19().to(device).eval()

# Hyperparameters
num_epochs = 101
alpha = 1
beta = 1e6
style_weights = {'conv1_1': 1., 'conv2_1': 0.75, 'conv3_1': 0.2, 'conv4_1': 0.2, 'conv5_1': 0.2}

with torch.no_grad():
    content_features = model(content_image)
    style_features = model(style_image)
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# Training loop
start_time = time.time()  # Record start time
for epoch in range(num_epochs):

    # Forward pass
    target_features = model(target_tensor)

    # Compute content loss
    content_loss_value = content_loss(content_features['conv4_2'], target_features['conv4_2'])

    # Compute style loss
    style_loss_value = sum(style_loss(style_features[layer], target_features[layer]) * style_weights[layer] for layer in style_weights.keys())

    # Total loss
    total_loss = alpha * content_loss_value + beta * style_loss_value

    # Optimization
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # Print loss and visualize image
    if (epoch + 1) % num_epochs == 0:
      end_time = time.time()  # Record end time
      epoch_time = end_time - start_time  # Calculate epoch time
      print(f'Epoch [{epoch+1}/{num_epochs}], Total Loss: {total_loss.item()}, Time: {epoch_time:.2f} seconds')
      plt.figure()
      imshow(target_tensor, title=f'Epoch {epoch+1}')
      plt.show()