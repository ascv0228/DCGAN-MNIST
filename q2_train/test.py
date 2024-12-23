import random
import torch
import os
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt

from hyperparameters import *
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
D = Discriminator(ngpu=1).eval()
G = Generator(ngpu=1).eval()

base_folder = f"./weights_{ngf}"

# load weights
D.load_state_dict(torch.load(f'{base_folder}/netD_epoch_29.pth'))
G.load_state_dict(torch.load(f'{base_folder}/netG_epoch_29.pth'))
if torch.cuda.is_available():
    D = D.cuda()
    G = G.cuda()

batch_size = 64
latent_size = 100
fixed_noise = torch.randn(batch_size, latent_size, 1, 1)

if torch.cuda.is_available():
    fixed_noise = fixed_noise.cuda()
fake_images = G(fixed_noise)

fake_images_np = fake_images.cpu().detach().numpy()
fake_images_np = fake_images_np.reshape(fake_images_np.shape[0], ngf, ngf)
R, C = 8, 8
plt.figure(figsize=(R, C))
for i in range(batch_size):
    plt.subplot(R, C, i + 1)
    plt.imshow(fake_images_np[i], cmap='gray')
    plt.axis('off')

plt.subplots_adjust(wspace=0, hspace=0)
plt.margins(0, 0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())

plt.show()