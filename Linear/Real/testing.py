import os
import torch
import torch.nn as nn
from torchvision.utils import save_image, make_grid

#Generator
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Tanh()
        )
    def forward(self, z):
        return self.model(z)

#Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
            
        )
    def forward(self, img):
        return self.model(img)

def generate_images(generator, latent_dim, a, b):
    generator.eval()

    batch_size = a * b
    z = torch.randn(batch_size, latent_dim).to(device)

    with torch.no_grad():
        generated_images = generator(z).view(batch_size, 1, 28, 28)

    grid = make_grid(generated_images, nrow=b, normalize=True, pad_value=1)
    output_folder = 'generated_images'
    os.makedirs(output_folder, exist_ok=True)
    save_image(grid, os.path.join(output_folder, 'grid_image.png'))


if __name__ == "__main__":
    latent_dim = 100
    
    a = int(input("Enter dimensions\nEnter the number of rows (a): "))
    b = int(input("Enter the number of columns (b): "))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    generator = Generator(latent_dim).to(device)
    discriminator = Discriminator().to(device)
    generator.load_state_dict(torch.load('generator.pth', map_location=torch.device(device)))
    discriminator.load_state_dict(torch.load('discriminator.pth', map_location=torch.device(device)))

    generate_images(generator, latent_dim, a, b)

