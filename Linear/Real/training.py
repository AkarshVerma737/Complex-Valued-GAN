import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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

#Training
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

            #Discriminator training
            optimizer_D.zero_grad()
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_imgs = generator(z)

            real_imgs_output = discriminator(real_imgs)
            fake_imgs_output = discriminator(fake_imgs.detach())

            real_loss = criterion(real_imgs_output, real_labels)
            fake_loss = criterion(fake_imgs_output, fake_labels)

            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_D.step()

            #Generator training
            optimizer_G.zero_grad()
            fake_imgs_output = discriminator(fake_imgs)
            g_loss = criterion(fake_imgs_output,real_labels)
            g_loss.backward()
            optimizer_G.step()

            if i % 400 == 0:
                print(f"Epoch [{epoch}/{num_epochs}] Batch [{i}/{len(dataloader)}], D_Loss: {d_loss.item()}, G_Loss: {g_loss.item()}")

    #Save model checkpoints
    torch.save(generator.state_dict(), 'generator.pth')
    torch.save(discriminator.state_dict(), 'discriminator.pth')

#Main function
if __name__ == "__main__":
    latent_dim = 100
    batch_size = 64
    num_epochs = 200
    lr = 0.0002

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    mnist_data = datasets.MNIST(root='data', train=True, transform=transform, download=True)
    dataloader = DataLoader(mnist_data, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    generator = Generator(latent_dim).to(device)
    discriminator = Discriminator().to(device)

    train_gan(generator, discriminator, dataloader, latent_dim, num_epochs, lr)