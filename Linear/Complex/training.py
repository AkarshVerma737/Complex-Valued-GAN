import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import activation as an

# Generator
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256, dtype=torch.complex64),
            an.Swish(),
            nn.Linear(256, 512, dtype=torch.complex64),
            an.Swish(),
            nn.Linear(512, 784, dtype=torch.complex64),
            an.CTanh()
        )

    def forward(self, z):
        return self.model(z)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img)

# Create image
def create_Image(output):
    real_part = torch.clamp(torch.real(output), -1.0, 1.0)
    imag_part = torch.clamp(torch.imag(output), -1.0, 1.0)
    return real_part

# Diversity loss function
def diversity_loss(generator, z1, z2):
    G_z1 = generator(z1)
    G_z2 = generator(z2)
    
    dI = torch.norm(G_z1 - G_z2)
    dz = torch.norm(z1 - z2)
    
    return (1/dI)

# Training function
def train_gan(generator, discriminator, dataloader, latent_dim, num_epochs, lr):
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=lr)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            batch_size = imgs.size(0)
            
            real_imgs = imgs.view(batch_size, -1).to(device)
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # Discriminator training
            optimizer_D.zero_grad()
            z = torch.randn(batch_size, latent_dim, dtype=torch.complex64).to(device)
            fake_imgs = generator(z)
            fake_output = discriminator(create_Image(fake_imgs))

            if torch.isnan(fake_output).any():
                print("NAN D")
                continue

            real_loss = criterion(discriminator(real_imgs), real_labels)
            fake_loss = criterion(fake_output, fake_labels)
            
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_D.step()

            # Generator training
            optimizer_G.zero_grad()
            fake_imgs = generator(z)
            g_loss = criterion(discriminator(create_Image(fake_imgs)), real_labels)
            
            # Diversity loss
            if(epoch>30):
                z1 = torch.randn(batch_size, latent_dim, dtype=torch.complex64).to(device)
                z2 = torch.randn(batch_size, latent_dim, dtype=torch.complex64).to(device)
                div_loss = diversity_loss(generator, z1, z2)
            else:
                div_loss=0
            
            total_g_loss = g_loss + div_loss
            total_g_loss.backward()
            optimizer_G.step()

            if i % 100 == 0:
                print(f"Epoch [{epoch}/{num_epochs}] Batch [{i}/{len(dataloader)}], D_Loss: {d_loss.item()}, G_Loss: {total_g_loss.item()}")

        if epoch % 10 == 0:
            torch.save(generator.state_dict(), f'generator_{epoch}.pth')
            torch.save(discriminator.state_dict(), f'discriminator_{epoch}.pth')

    # Save model checkpoints
    torch.save(generator.state_dict(), 'generator.pth')
    torch.save(discriminator.state_dict(), 'discriminator.pth')

# Main function
if __name__ == "__main__":
    latent_dim = 100
    batch_size = 64
    num_epochs = 200
    lr = 0.0002

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    mnist_data = datasets.MNIST(root='data', train=True, transform=transform, download=True)
    dataloader = DataLoader(mnist_data, batch_size=batch_size, shuffle=True)
    generator = Generator(latent_dim).to(device)
    discriminator = Discriminator().to(device)

    train_gan(generator, discriminator, dataloader, latent_dim, num_epochs, lr)
