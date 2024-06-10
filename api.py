from flask import Flask, request, jsonify, send_file
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import time
import io

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def img_loader(img_path, size):
    image = Image.open(img_path).convert('RGB')
    loader = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = loader(image)[:3, :, :].unsqueeze(0)
    return image.to(device, torch.float)

class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        self.chosen_features = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '21': 'conv4_2',
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

def content_loss(target_features, generated_features):
    return torch.mean((target_features - generated_features) ** 2)

def gram_matrix(input_tensor):
    b, c, h, w = input_tensor.size()
    features = input_tensor.view(b * c, h * w)
    return torch.mm(features, features.t()) / (b * c * h * w)

def style_loss(target_features, generated_features):
    target_gram = gram_matrix(target_features)
    generated_gram = gram_matrix(generated_features)
    return torch.mean((target_gram - generated_gram) ** 2)

@app.route('/style_transfer', methods=['POST'])
def style_transfer():
    content_file = request.files['content']
    style_file = request.files['style']
    size = int(request.form.get('size', 356))
    num_epochs = int(request.form.get('num_epochs', 101))
    alpha = float(request.form.get('alpha', 1))
    beta = float(request.form.get('beta', 1e6))

    content_image = img_loader(content_file, size)
    style_image = img_loader(style_file, size)
    target_tensor = content_image.clone().requires_grad_(True)

    optimizer = optim.Adam([target_tensor], lr=0.1)
    model = VGG19().to(device).eval()

    with torch.no_grad():
        content_features = model(content_image)
        style_features = model(style_image)

    style_weights = {'conv1_1': 1., 'conv2_1': 0.75, 'conv3_1': 0.2, 'conv4_1': 0.2, 'conv5_1': 0.2}

    for _ in range(num_epochs):
        target_features = model(target_tensor)
        content_loss_value = content_loss(content_features['conv4_2'], target_features['conv4_2'])
        style_loss_value = sum(style_loss(style_features[layer], target_features[layer]) * style_weights[layer] for layer in style_weights.keys())
        total_loss = alpha * content_loss_value + beta * style_loss_value

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    image = target_tensor.cpu().clone().detach()
    image = image.numpy().squeeze(0)
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    image = (image * 255).astype('uint8')
    image = Image.fromarray(image)

    img_io = io.BytesIO()
    image.save(img_io, 'PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
