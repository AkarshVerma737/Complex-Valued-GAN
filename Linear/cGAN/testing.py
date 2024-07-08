import os
import torch
import torch.nn as nn
from torchvision.utils import save_image, make_grid
import activation as an
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Generator
class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256, dtype=torch.complex64),
            an.Swish(),
            nn.Linear(256, 512, dtype=torch.complex64),
            an.Swish(),
            nn.Linear(512, 784, dtype=torch.complex64),
            an.CTanhR()
        )

    def forward(self, z, labels):
        clabels = torch.complex(labels, torch.zeros_like(labels)).to(device)
        gen_input = torch.cat((z, clabels), dim=1)
        return self.model(gen_input)

def generate_images(generator, latent_dim, num_classes, a, b):
    generator.eval()

    batch_size = a * b
    z = torch.randn(batch_size, latent_dim, dtype=torch.complex64).to(device)
    labels = torch.randint(0, num_classes, (batch_size,))
    labels = torch.eye(10)[labels]

    with torch.no_grad():
        generated_images = (generator(z, labels).real).view(batch_size, 1, 28, 28)

    grid = make_grid(generated_images, nrow=b, normalize=True, pad_value=1)
    output_folder = 'generated_images'
    os.makedirs(output_folder, exist_ok=True)
    save_image(grid, os.path.join(output_folder, 'grid_image.png'))


if __name__ == "__main__":
    latent_dim = 100
    num_classes = 10  # Number of classes in MNIST dataset
    a = int(input("Enter the number of rows (a): "))
    b = int(input("Enter the number of columns (b): "))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator(latent_dim, num_classes).to(device)
    generator.load_state_dict(torch.load('generator.pth', map_location=device))

    generate_images(generator, latent_dim, num_classes, a, b)
