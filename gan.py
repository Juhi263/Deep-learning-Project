import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
image_size = 28 * 28
latent_size = 100
hidden_size = 256
batch_size = 500
num_epochs = 20
learning_rate = 0.0002

# MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Discriminator network
class Discriminator(nn.Module):
    def __init__(self, image_size, hidden_size):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(image_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Generator network
class Generator(nn.Module):
    def __init__(self, latent_size, hidden_size, image_size):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, image_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Initialize networks
discriminator = Discriminator(image_size, hidden_size).to(device)
generator = Generator(latent_size, hidden_size, image_size).to(device)

# Loss function and optimizer
criterion = nn.BCELoss()
d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate)
g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate)

# Function to generate samples
def generate_samples(generator, num_samples):
    latent = torch.randn(num_samples, latent_size).to(device)
    with torch.no_grad():
        generated_images = generator(latent)
    return generated_images.view(num_samples, 1, 28, 28)

# Training the GAN
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(train_loader):
        images = images.reshape(batch_size, -1).to(device)

        # Labels for real and fake images
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Training the discriminator
        discriminator.zero_grad()
        real_outputs = discriminator(images)
        d_loss_real = criterion(real_outputs, real_labels)
        d_loss_real.backward()

        latent = torch.randn(batch_size, latent_size).to(device)
        fake_images = generator(latent)
        fake_outputs = discriminator(fake_images.detach())
        d_loss_fake = criterion(fake_outputs, fake_labels)
        d_loss_fake.backward()

        d_loss = d_loss_real + d_loss_fake
        d_optimizer.step()

        # Training the generator
        generator.zero_grad()
        fake_outputs = discriminator(fake_images)
        g_loss = criterion(fake_outputs, real_labels)
        g_loss.backward()
        g_optimizer.step()

        if (i+1) % 200 == 0:
            print('Epoch [{}/{}], Step [{}/{}], D Loss: {:.4f}, G Loss: {:.4f}'
                  .format(epoch, num_epochs, i+1, total_step, d_loss.item(), g_loss.item()))

    # Save generated images
    if (epoch+1) % 10 == 0:
        torchvision.utils.save_image(generate_samples(generator, 16), 'generated_images_{}.png'.format(epoch+1))

# Visualize generated samples
plt.figure(figsize=(10, 10))
plt.imshow(np.transpose(torchvision.utils.make_grid(generate_samples(generator, 64), padding=2, normalize=True).cpu(), (1, 2, 0)))
plt.axis('off')
plt.show()
