import os
import torch
import torch.nn as nn
from torchvision.utils import save_image, make_grid
import activation as an
#Generator
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256, dtype = torch.complex64),
            an.Switch(),
            nn.Linear(256, 512, dtype = torch.complex64),
            an.Switch(),
            nn.Linear(512, 784,dtype = torch.complex64),
            an.CTanhR()
        )
    def forward(self, z):
        return self.model(z)



def generate_images(generator, latent_dim, a, b):
    generator.eval()

    batch_size = a * b
    z = torch.randn(batch_size, latent_dim, dtype = torch.complex64).to(device)

    with torch.no_grad():
        generated_images = (generator(z).real).view(batch_size, 1, 28, 28)

    grid = make_grid(generated_images, nrow=b, normalize=True, pad_value=1)
    output_folder = 'generated_images'
    os.makedirs(output_folder, exist_ok=True)
    save_image(grid, os.path.join(output_folder, 'grid_image.png'))


if __name__ == "__main__":
    latent_dim = 100
    st = 'Y'
    while (st == 'Y'):
        a = int(input("Enter dimensions\nEnter the number of rows (a): "))
        b = int(input("Enter the number of columns (b): "))

        device = torch.device("cuda")# if torch.cuda.is_available() else "cpu")
        #device = "cpu"
        generator = Generator(latent_dim).to(device)
        #generator.load_state_dict(torch.load('generatori.pth', map_location=torch.device('cpu')))
        generator.load_state_dict(torch.load('generator 20.pth'))
        generate_images(generator, latent_dim, a, b)
        st = (input("Generate Random Handwritten Numbers? Y/N: "))
