import os
import torch
import torch.nn as nn
from torchvision.utils import save_image, make_grid
import activation as an
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import complexconv as cn

# Generator
class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes, ngf, nc):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim + num_classes, ngf * 4, 4, 1, 0 , dtype = torch.complex64),
            cn.ComplexBatchNorm2d(ngf * 4),
            an.CReLUR(),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1 , dtype = torch.complex64),
            cn.ComplexBatchNorm2d(ngf * 2),
            an.CReLUR(),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1 , dtype = torch.complex64),
            cn.ComplexBatchNorm2d(ngf),
            an.CReLUR(),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1 , dtype = torch.complex64),
            an.CTanhR()
        )
        

    def forward(self, noise, labels):
        labels = self.label_emb(labels).unsqueeze(2).unsqueeze(3)
        labels = labels.expand(labels.size(0), labels.size(1), noise.size(2), noise.size(3))
        input = torch.cat((noise, labels), 1)
        return self.model(input)

def generate_images(generator, latent_dim, num_classes, a, b):
    generator.eval()

    batch_size = a * b
    z = torch.randn(batch_size, latent_dim, 1, 1,dtype = torch.complex64).to(device)
    labels = torch.randint(0, num_classes, (batch_size,)).to(device)

    with torch.no_grad():
        generated_images = (generator(z, labels).real).view(batch_size, 3, 32, 32)

    grid = make_grid(generated_images, nrow=b, normalize=True, pad_value=1)
    output_folder = 'generated_images'
    os.makedirs(output_folder, exist_ok=True)
    save_image(grid, os.path.join(output_folder, 'grid_image.png'))


if __name__ == "__main__":
    latent_dim = 100
    num_classes = 10
    nc = 3
    ngf = 64
    a = int(input("Enter the number of rows (a): "))
    b = int(input("Enter the number of columns (b): "))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator(latent_dim, num_classes,ngf,nc).to(device)
    generator.load_state_dict(torch.load('generator_1.pth', map_location=device))

    generate_images(generator, latent_dim, num_classes, a, b)
