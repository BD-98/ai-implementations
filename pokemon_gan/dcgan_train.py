import torch
import torch.nn as nn 
from data_utils import data, device
from dcgan_models import Generator, Discriminator, latent_size
from tqdm import trange

# Models and losses 
g = Generator().to(device)
d = Discriminator().to(device)
g_optim = torch.optim.Adam(g.parameters())
d_optim = torch.optim.Adam(d.parameters())
loss_function = nn.BCELoss()

# Training Loop 
for _ in trange(15):
    for poke_images,_ in data:
        #=====================
        # Discriminator 
        #=====================
        bs = len(poke_images)
        x_real = poke_images.to(device)
        x_fake = torch.randn(bs, latent_size, 1, 1).to(device)
        y_real = torch.ones(bs, 1).to(device)
        y_fake = torch.zeros(bs, 1).to(device)

        # Forward Pass 
        real_out = d(x_real)
        fake_out = d(g(x_fake))
        real_loss = loss_function(real_out, y_real)
        fake_loss = loss_function(fake_out, y_fake)
        d_loss = real_loss + fake_loss
        
        # Update Weights 
        d_optim.zero_grad()
        d_loss.backward()
        d_optim.step()
        
        #===================
        # Generator 
        #===================
        x_fake = torch.randn(bs, latent_size, 1, 1).to(device)
        y_real = torch.ones(bs, 1).to(device)

        # Forward Pass 
        fake_pokemons = g(x_fake)
        fake_out = d(fake_pokemons)
        g_loss = loss_function(fake_out, y_real)

        # Update Weights 
        g_optim.zero_grad()
        g_loss.backward()
        g_optim.step()
    
    # Display Losses 
    print(f"d_losses = {d_loss.item():.3f}, g_losses = {g_loss.item():.3f}")




