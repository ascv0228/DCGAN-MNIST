from __future__ import print_function
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils

from hyperparameters import *
from custom_dataset import CustomDataset

if ngf == 64:
    from network_64 import Generator, Discriminator, weights_init
else:
    from network_28 import Generator, Discriminator, weights_init
cudnn.benchmark = True

#set manual seed to a constant get a consistent output
manualSeed = random.randint(1, 10000)
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

#loading the dataset
DATASET_PATH        = "../../cvdl_hw2_data/Q2_images/data/mnist"
dataset = CustomDataset(root_dir=DATASET_PATH,transform=transforms.Compose([
                            transforms.Resize(ndf),
                            transforms.RandomRotation(60),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,)),
                        ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

#checking the availability of cuda devices
device = 'cuda' if torch.cuda.is_available() else 'cpu'

base_folder = f"./weights_{ngf}"
if not os.path.exists(base_folder):
    os.makedirs(base_folder)

image_folder = f"./output_{ngf}"
if not os.path.exists(image_folder):
    os.makedirs(image_folder)


netG = Generator(ngpu).to(device)
netG.apply(weights_init)
# netG.load_state_dict(torch.load('weights/netG_epoch_99.pth'))
print(netG)


netD = Discriminator(ngpu).to(device)
netD.apply(weights_init)
# netD.load_state_dict(torch.load('weights/netD_epoch_99.pth'))
print(netD)

criterion = nn.BCELoss()

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

fixed_noise = torch.randn(64, nz, 1, 1, device=device)
real_label = 1
fake_label = 0

niter = 30

G_losses = []
D_losses = []

# Commented out IPython magic to ensure Python compatibility.
for epoch in range(niter):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)

        output = netD(real_cpu)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # train with fake
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label) 
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, niter, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        if i % 100 == 0:
            vutils.save_image(real_cpu,f'{image_folder}/real_samples.png' ,normalize=True)
            fake = netG(fixed_noise)
            vutils.save_image(fake.detach(),f'{image_folder}/fake_samples_epoch_{epoch:0>2}.png', normalize=True)        
        
        G_losses.append(errG.item())
        D_losses.append(errD.item())
    torch.save(netG.state_dict(), f'{base_folder}/netG_epoch_{epoch:0>2}.pth')
    torch.save(netD.state_dict(), f'{base_folder}/netD_epoch_{epoch:0>2}.pth')
