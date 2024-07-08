import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Generator
class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes, ngf, nc):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim + num_classes, ngf * 4, 4, 1, 0),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        labels = self.label_emb(labels).unsqueeze(2).unsqueeze(3)
        labels = labels.expand(labels.size(0), labels.size(1), noise.size(2), noise.size(3))
        input = torch.cat((noise, labels), 1)
        return self.model(input)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, num_classes, ndf, nc):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            
            nn.Conv2d(nc + num_classes, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        labels = self.label_emb(labels).unsqueeze(2).unsqueeze(3)
        labels = labels.expand(labels.size(0), labels.size(1), img.size(2), img.size(3))
        input = torch.cat((img, labels), 1) 
        return self.model(input).view(-1, 1)


# Training function
def train_gan(generator, discriminator, dataloader, latent_dim, num_epochs, lr, device):
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=lr)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for i, (imgs, labels) in enumerate(dataloader):
            batch_size = imgs.size(0)
            real_imgs = imgs.to(device)
            labels = labels.to(device)
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # Train Discriminator
            discriminator.zero_grad()
            z = torch.randn(batch_size, latent_dim, 1, 1).to(device)
            fake_imgs = generator(z, labels)
            real_output = discriminator(real_imgs, labels)
            fake_output = discriminator(fake_imgs.detach(), labels)
            real_loss = criterion(real_output, real_labels)
            fake_loss = criterion(fake_output, fake_labels)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            generator.zero_grad()
            fake_output = discriminator(fake_imgs, labels)
            g_loss = criterion(fake_output, real_labels)
            g_loss.backward()
            optimizer_G.step()

            if i % 100 == 0:
                print(f"Epoch [{epoch}/{num_epochs}] Batch [{i}/{len(dataloader)}], D_Loss: {d_loss.item()}, G_Loss: {g_loss.item()}")

        if epoch % 10 == 0:
            torch.save(generator.state_dict(), f'generator_{epoch}.pth')
            torch.save(discriminator.state_dict(), f'discriminator_{epoch}.pth')

    # Save final models
    torch.save(generator.state_dict(), 'generator.pth')
    torch.save(discriminator.state_dict(), 'discriminator.pth')

# Main function
if __name__ == "__main__":
    latent_dim = 100
    batch_size = 128
    num_epochs = 200
    nc = 3
    ngf = 64
    ndf = 64
    lr = 0.0002
    num_classes = 10

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cifar_data = datasets.CIFAR10(root='data', train=True, transform=transform, download=True)
    dataloader = DataLoader(cifar_data, batch_size=batch_size, shuffle=True)
    generator = Generator(latent_dim, num_classes, ngf, nc).to(device)
    discriminator = Discriminator(num_classes, ndf, nc).to(device)

    train_gan(generator, discriminator, dataloader, latent_dim, num_epochs, lr, device)
