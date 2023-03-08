import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


def get_generator_block(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU(inplace=True)
    )


class Generator(nn.Module):
    def __init__(self, z_dim=10, im_dim=784, hidden_dim=128):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            get_generator_block(z_dim, hidden_dim),
            get_generator_block(hidden_dim, hidden_dim * 2),
            get_generator_block(hidden_dim * 2, hidden_dim * 4),
            get_generator_block(hidden_dim * 4, hidden_dim * 8),
            nn.Linear(hidden_dim * 8, im_dim),
            nn.Sigmoid()
        )

    def forward(self, noise):
        return self.gen(noise)


def get_noise(n_samples, z_dim, device='cpu'):
    return torch.randn(n_samples, z_dim, device=device)


def get_discriminator_block(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.LeakyReLU(negative_slope=0.2)
    )


class Discriminator(nn.Module):
    def __init__(self, im_dim=784, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            get_discriminator_block(im_dim, hidden_dim * 4),
            get_discriminator_block(hidden_dim * 4, hidden_dim * 2),
            get_discriminator_block(hidden_dim * 2, hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, image):
        return self.disc(image)


def get_disc_loss(gen, disc, criterion, real, num_images, z_dim, device):
    disc_opt.zero_grad()
    gen_opt.zero_grad()
    noise = get_noise(num_images, z_dim, device)
    generated_images = gen(noise).detach()

    y_fake = disc(generated_images)
    loss_fake = criterion(y_fake, torch.zeros_like(y_fake))

    y_real = disc(real)
    loss_real = criterion(y_real, torch.ones_like(y_real))

    disc_loss = (loss_fake + loss_real) / 2
    return disc_loss


def get_gen_loss(gen, disc, criterion, num_images, z_dim, device):
    noise = get_noise(num_images, z_dim, device)
    generated_images = gen(noise)

    y_pred = disc(generated_images)
    gen_loss = criterion(y_pred, torch.ones_like(y_pred))
    return gen_loss


if __name__ == "__main__":
    criterion = nn.BCEWithLogitsLoss()
    n_epochs = 200
    z_dim = 64
    display_step = 469*5
    batch_size = 128
    device = 'cpu'
    dataloader = DataLoader(
        MNIST('.', download=False, transform=transforms.ToTensor()),
        batch_size=batch_size,
        shuffle=True)

    gen = Generator(z_dim).to(device)
    gen_opt = torch.optim.Adam(gen.parameters())
    disc = Discriminator().to(device)
    disc_opt = torch.optim.Adam(disc.parameters())


    cur_step = 0
    mean_generator_loss = 0
    mean_discriminator_loss = 0
    test_generator = True
    gen_loss = False
    error = False
    for epoch in range(n_epochs):
        for real, _ in tqdm(dataloader):
            cur_batch_size = len(real)
            # Flatten the batch of real images from the dataset
            real = real.view(cur_batch_size, -1).to(device)

            # Update discriminator
            disc_opt.zero_grad()
            disc_loss = get_disc_loss(gen, disc, criterion, real, cur_batch_size, z_dim, device)
            disc_loss.backward(retain_graph=True)
            disc_opt.step()

            # Update generator
            gen_opt.zero_grad()
            gen_loss = get_gen_loss(gen, disc, criterion, cur_batch_size, z_dim, device)
            gen_loss.backward(retain_graph=True)
            gen_opt.step()

            # Keep track of the average losses
            mean_discriminator_loss += disc_loss.item() / display_step
            mean_generator_loss += gen_loss.item() / display_step

            ### Visualization code
            if cur_step % display_step == 0 and cur_step > 0:
                print(f'\nStep {cur_step}: Generator loss: {mean_generator_loss}, '
                      f'discriminator loss: {mean_discriminator_loss}')
                fake_noise = get_noise(cur_batch_size, z_dim, device)
                fake = gen(fake_noise)
                show_tensor_images(fake)
                mean_generator_loss = 0
                mean_discriminator_loss = 0
            cur_step += 1
