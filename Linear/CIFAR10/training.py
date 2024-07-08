import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import activation as an

# Generator
class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256, dtype=torch.complex64),
            an.Swish(),
            nn.Linear(256, 512, dtype=torch.complex64),
            an.Swish(),
            nn.Linear(512, 1024, dtype=torch.complex64),
            an.Swish(),
            nn.Linear(1024, 2048, dtype=torch.complex64),
            an.Swish(),
            nn.Linear(2048, 32*32*3, dtype=torch.complex64),
            an.CTanhR()
        )

    def forward(self, z, labels):
        clabels = torch.complex(labels,torch.zeros_like(labels)).to(device)
        gen_input = torch.cat((z, clabels), dim=1)
        return self.model(gen_input)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, num_classes):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(32*32*3 + num_classes, 2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        disc_input = torch.cat((img, labels.to(device)), dim=1)
        return self.model(disc_input)

# Create image
def create_Image(output):
    real_part = torch.clamp(torch.real(output), min=-0.999999, max=0.999999)
    imag_part = torch.clamp(torch.imag(output), min=-0.999999, max=0.999999)
    return real_part

# Diversity Loss
def diversity_loss(generator, z1, z2,labels):
    dz = torch.sqrt((z1.real - z2.real) ** 2 + (z1.imag - z2.imag) ** 2)
    Dz = torch.mean(dz).item()
    
    Gz1 = generator(z1,labels)
    Gz2 = generator(z2,labels)

    relu = nn.ReLU()
    Gz1_relu = relu(Gz1.real)
    Gz2_relu = relu(Gz2.real)

    Di = torch.sqrt((Gz1_relu - Gz2_relu) ** 2)
    DI = torch.sum(Di).item()
    diversity = 1 / DI
    return diversity

# Training function
def train_gan(generator, discriminator, dataloader, latent_dim, num_epochs, lr):
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=lr)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for i, (imgs, labels) in enumerate(dataloader):
            batch_size = imgs.size(0)
            labels = torch.eye(10)[labels]
            real_imgs = imgs.view(batch_size, -1).to(device)
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # Discriminator training
            optimizer_D.zero_grad()
            z = torch.randn(batch_size, latent_dim, dtype=torch.complex64).to(device)
            fake_imgs = generator(z, labels)
            Image = create_Image(fake_imgs).to(device)
            fake_output = torch.clamp(discriminator(Image, labels), min=0.000001, max=0.999999)
            if torch.isnan(fake_output).any():
                print("NAN D")
                continue
            real_loss = criterion(torch.clamp(discriminator(real_imgs, labels), min=0.000001, max=0.999999), real_labels)
            fake_loss = criterion(fake_output, fake_labels)
            
            d_loss = real_loss + fake_loss
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 3.0)
            optimizer_D.step()

            # Generator training
            optimizer_G.zero_grad()
            fake_imgs = generator(z, labels)
            Image = create_Image(fake_imgs)
            fake_output = torch.clamp(discriminator(Image, labels), min=0.000001, max=0.999999)
            if torch.isnan(fake_output).any():
                print("NAN G")
                continue
            g_loss = criterion(fake_output, real_labels)
            
            # Diversity loss
            #if(epoch > 30):
            #    z1 = torch.randn(batch_size, latent_dim, dtype=torch.complex64).to(device)
            #    z2 = torch.randn(batch_size, latent_dim, dtype=torch.complex64).to(device)
            #    div_loss = diversity_loss(generator, z1, z2,labels)
            #:
            div_loss = 0
            
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
    lr = 0.00005
    num_classes = 10

    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cifar_data = datasets.CIFAR10(root='data', train=True, transform=transform, download=True)
    dataloader = DataLoader(cifar_data, batch_size=batch_size, shuffle=True)
    generator = Generator(latent_dim, num_classes).to(device)
    discriminator = Discriminator(num_classes).to(device)

    train_gan(generator, discriminator, dataloader, latent_dim, num_epochs, lr)
