import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
latent_size = 100
hidden_size = 256
image_size = 784
num_epochs = 100
batch_size = 100
sample_dir = 'samples'

# Create a directory if not exists
import os
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

# Image processing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data/',
                                           train=True,
                                           transform=transform,
                                           download=True)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

# Generator network
class Generator(nn.Module):
    def __init__(self, latent_size, hidden_size, image_size):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_size + 10, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, image_size),
            nn.Tanh()
        )
        
    def forward(self, x, y):
        print(x)
        print(y)
        x = torch.cat([x, y], 1)
        print(x)
        return self.fc(x)

# Discriminator network
class Discriminator(nn.Module):
    def __init__(self, image_size, hidden_size):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(image_size + 10, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, y):
        x = torch.cat([x, y], 1)
        return self.fc(x)

# Initialize networks
generator = Generator(latent_size, hidden_size, image_size).to(device)
discriminator = Discriminator(image_size, hidden_size).to(device)

# Loss function
criterion = nn.BCELoss()

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Training
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(batch_size, -1).to(device)
        labels = torch.eye(10)[labels].to(device)  # One-hot encode labels
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        
        # Train discriminator
        optimizer_D.zero_grad()
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = generator(z, labels)
        d_real = discriminator(images, labels)
        d_fake = discriminator(fake_images, labels)
        d_loss_real = criterion(d_real, real_labels)
        d_loss_fake = criterion(d_fake, fake_labels)
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()
        
        # Train generator
        optimizer_G.zero_grad()
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = generator(z, labels)
        d_fake = discriminator(fake_images, labels)
        g_loss = criterion(d_fake, real_labels)
        g_loss.backward()
        optimizer_G.step()
        
        if (i+1) % 200 == 0:
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}' 
                  .format(epoch, num_epochs, i+1, total_step, d_loss.item(), g_loss.item()))
    
    # Save generated images
    if (epoch+1) == 1:
        torchvision.utils.save_image(fake_images.data[:25], 
                                     os.path.join(sample_dir, 'fake_images-{}.png'.format(epoch+1)), 
                                     nrow=5, normalize=True)

# Save the model checkpoints
torch.save(generator.state_dict(), 'generator.ckpt')
torch.save(discriminator.state_dict(), 'discriminator.ckpt')
