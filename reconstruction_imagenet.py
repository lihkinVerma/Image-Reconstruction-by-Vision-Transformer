#--------------------------------------------------------------------
# Importing required libraries
#--------------------------------------------------------------------
import random
import numpy as np
import pandas as pd
import PIL.Image as Image
from glob import glob
from mymodels import VisionTransformer, ReconNet
from mymodels.unet import Unet
from mymodels.discriminatorv2 import Discriminator
import matplotlib.pyplot as plt
from myutils import imshow
from datetime import datetime


# importing pytorch functions
import torch
from torchvision.utils import make_grid
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as tvtransforms
from torch.nn import SmoothL1Loss, BCELoss

# importing utils required in th code
from utils import subsample
from utils import transforms
from utils.evaluate import ssim, psnr, nmse
from utils.losses import SSIMLoss

# Device
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cuda'
np.random.seed(42)
random.seed(42)

print("******* STARTED AT ************", datetime.now())

#--------------------------------------------------------------------
# Loading dataset
#--------------------------------------------------------------------
class ImagenetDataset(Dataset):
    def __init__(self, isval=False):

        if isval:
            ## combine paths of each imagenet validation image into a single list
            self.files = []
            pattern = "*.JPEG"
            for dir, _, _ in os.walk('./Data/tiny-imagenet-200/test/images/'):
                self.files.extend(glob(os.path.join(dir, pattern)))
        else:
            ## combine paths of each imagenet training image into a single list
            self.files = []  # get path of each imagenet images
            pattern = "*.JPEG"
            for dir, _, _ in os.walk('./Data/tiny-imagenet-200/train/'):
                self.files.extend(glob(os.path.join(dir, pattern)))

        ##################### Change image size ########################
        ############## for best model, use tvtransforms.Resize(64*3,), tvtransforms.RandomCrop(60*3),
        self.transform = tvtransforms.Compose([
            tvtransforms.Resize(64*3, ), # original 320
            tvtransforms.RandomCrop(60*3), # original 272
            tvtransforms.Grayscale(1),
            tvtransforms.RandomVerticalFlip(p=0.5),
            tvtransforms.RandomHorizontalFlip(p=0.5),
            tvtransforms.ToTensor(),
        ])
        ################################################################

        self.factors = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    def get_mask_func(self, samp_style, factor, ):
        center_fractions = 0.08 * 4 / factor
        if samp_style == 'random':
            mask_func = subsample.RandomMaskFunc(
                center_fractions=[center_fractions],
                accelerations=[factor],
            )
        elif samp_style == 'equidist':
            mask_func = subsample.EquispacedMaskFunc(
                center_fractions=[center_fractions],
                accelerations=[factor],
            )
        return mask_func

    def add_gaussian_noise(self, x):
        ch, row, col = x.shape
        mean = 0
        var = 0.05
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (ch, row, col))
        gauss = gauss.reshape(ch, row, col)
        noisy = x + gauss
        return noisy.float()

    def __len__(self, ):

        return len(self.files)

    def __getitem__(self, idx):
        image = Image.open(self.files[idx]).convert("RGB")

        y = self.transform(image)

        if random.uniform(0, 1) < 0.5:
            y = torch.rot90(y, 1, [-2, -1])

        if random.uniform(0, 1) < 0.5:
            samp_style = 'random'
        else:
            samp_style = 'equidist'
        factor = random.choice(self.factors)
        mask_func = self.get_mask_func(samp_style, factor=5)
        masked_kspace, _ = transforms.apply_mask(y, mask_func)


        # masked_kspace = self.add_gaussian_noise(y)

        return (masked_kspace, y)

dataset = ImagenetDataset()
val_dataset = ImagenetDataset(isval=True)

# ntrain = len(dataset)
ntrain = 20000
train_dataset, _ = torch.utils.data.random_split(dataset, [ntrain, len(dataset) - ntrain],
                                                 generator=torch.Generator().manual_seed(42))
############## for best model, use batch_size = 45 (not sure)
batch_size = 35
epoch = 40
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True,
                         generator=torch.Generator().manual_seed(42))
valloader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=1, pin_memory=True,
                       generator=torch.Generator().manual_seed(42))

#--------------------------------------------------------------------
# Initialising Models
#--------------------------------------------------------------------
# Vision Transformer
############## for best model, use avrg_img_size = 180
avrg_img_size = 180
patch_size = 10
depth = 10
num_heads = 8
embed_dim = 44

net = VisionTransformer(
    avrg_img_size=avrg_img_size,
    patch_size=patch_size,
    in_chans=1, embed_dim=embed_dim,
    depth=depth, num_heads=num_heads,
    is_LSA=False, #---------------Parameter for adding LSA component
    is_SPT=False, #---------------Parameter for adding SPT component
    rotary_position_emb = True, #---------------Parameter for adding ROPE component
    use_pos_embed=False
)

#--------------- network for testing Unet architecture
# Unet - Uncomment for running the U-net Code
# net = Unet(
#     in_chans=1,
#     out_chans=1,
#     chans=64,
#     num_pool_layers=4,
#     )

#--------------------------------------------------------------------
# Creating a reconstruction network
#--------------------------------------------------------------------
model = ReconNet(net).to(device)

# Set biases to zero
for name, param in model.named_parameters():
    if name.endswith('.bias'):
        torch.nn.init.constant_(param, 0)
        param.requires_grad = False

print('#Params:', sum(p.numel() for p in model.parameters() if p.requires_grad))
print(model)

#--------------------------------------------------------------------
# Function to Save model
#--------------------------------------------------------------------
def save_model(path, model, train_hist, optimizer, scheduler=None):
    net = model.net
    if scheduler:
        checkpoint = {
            'model': ReconNet(net),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }
    else:
        checkpoint = {
            'model': ReconNet(net),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }

    torch.save(train_hist, path + 'train_hist.pt')
    torch.save(checkpoint, path + 'checkpoint.pth')


"""Choose optimizer"""
criterion = SSIMLoss().to(device)
optimizerG = optim.Adam(model.parameters(), lr=0.0)
train_hist = []
path = './'  # Path for saving model checkpoint and loss history
scheduler = optim.lr_scheduler.OneCycleLR(optimizerG, max_lr=0.0004,
                                          total_steps=epoch, pct_start=0.1,
                                          anneal_strategy='linear',
                                          cycle_momentum=False,
                                          base_momentum=0., max_momentum=0., div_factor=0.1 * epoch, final_div_factor=9)

#--------------------------------------------------------------------
# Start to train the model
#--------------------------------------------------------------------

#----------- discriminator parameters to test Adverserial loss addition to the network
run_discriminator = False
if run_discriminator:
    criterionGAN = BCELoss().to(device)
    # criterionGAN = GANLoss().to(device)
    # discriminator = PatchDiscriminator(input_nc=1).to(device)
    discriminator = Discriminator(in_channels = 1,
                                patch_size = patch_size,
                                extend_size = 2,
                                dim = 50,
                                blocks = depth,
                                num_heads = 8,
                                dim_head = None,
                                dropout = 0.1).to(device)
    optimizerD = torch.optim.Adam(discriminator.parameters(), lr=0.00005)

    gan_weight = 0.05
    ss_weight = 1
    schedulerD = optim.lr_scheduler.OneCycleLR(optimizerD, max_lr=0.00005,
                                          total_steps=epoch, pct_start=0.1,
                                          anneal_strategy='linear',
                                          cycle_momentum=False,
                                          base_momentum=0., max_momentum=0., div_factor=0.1 * epoch, final_div_factor=9)
#--------------------------------------------------------------------

if not run_discriminator:
    for epoch in range(0, epoch):  # loop over the dataset multiple times
        model.train()
        train_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, targets = data
            optimizerG.zero_grad()
            outputs = model(inputs.to(device))

            loss = criterion(outputs, targets.to(device), torch.tensor([1.], device=device))
            # loss = criterion(outputs, targets.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=1, error_if_nonfinite=True)
            optimizerG.step()

            train_loss += loss.item()

        scheduler.step()
        train_hist.append(train_loss / len(trainloader))
        save_model(path, model, train_hist, optimizerG, scheduler=scheduler)
        print('Epoch {}, Train loss.: {:0.4e}'.format(epoch + 1, train_hist[-1]))
else:
    for epoch in range(0, epoch):  # loop over the dataset multiple times
        model.train()
        losses_real = 0.0
        losses_fake = 0.0
        losses_ss = 0.0
        losses_gan = 0.0
        for i, data in enumerate(trainloader):
            inputs, targets = data
            target_len = len(targets)
            outputs = model(inputs.to(device))

            #######
            ## Train Discriminator
            #######
            optimizerD.zero_grad()

            pred_fake = discriminator.forward(outputs.detach())  # Detach to make sure no gradients go into generator
            loss_d_fake = criterionGAN(pred_fake.flatten(), torch.zeros(target_len).to(device))
            loss_d_fake.backward()

            pred_real = discriminator.forward(targets.to(device))
            loss_d_real = criterionGAN(pred_real.flatten(), torch.ones(target_len).to(device))
            loss_d_real.backward()

            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1, norm_type=1, error_if_nonfinite=True)
            optimizerD.step()

            losses_fake += loss_d_fake
            losses_real += loss_d_real

            #######
            ## Train generator
            #######
            optimizerG.zero_grad()
            loss_g_ss = criterion(outputs, targets.to(device), torch.tensor([1.], device=device))
            # loss_g_ss = criterion(outputs, targets.to(device))

            # Calculate adversarial loss
            pred_fake = discriminator.forward(outputs)
            loss_g_gan = criterionGAN(pred_fake.flatten(), torch.ones(target_len).to(device))

            loss_g = loss_g_ss * ss_weight + loss_g_gan * gan_weight

            loss_g.backward()
            losses_ss += loss_g_ss
            losses_gan += loss_g_gan
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=1, error_if_nonfinite=True)
            optimizerG.step()

        scheduler.step()
        schedulerD.step()
        trainloader_len = len(trainloader)
        train_hist.append((losses_ss.item() * ss_weight + loss_g_gan.item() * gan_weight)/trainloader_len)
        # save_model(path, model, train_hist, optimizerG, scheduler=scheduler)
        print('Epoch {}, Train loss. (generator): {:0.4e}, (discriminator: real, fake): {:0.4e}, {:0.4e}'.format(
            epoch + 1, train_hist[-1], losses_real.item()/trainloader_len, losses_fake.item()/trainloader_len))
        print('Epoch {}, Train loss. (SSIM, GAN): {:0.4e}, {:0.4e}'.format(
            epoch + 1, losses_ss.item()/trainloader_len, losses_gan.item()/trainloader_len))

# """Example reconstructions"""
# valloader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=0,
#                        generator=torch.Generator().manual_seed(1))
# dataiter = iter(valloader)
# model.eval()
# maxval = torch.tensor([1.], device='cpu')
# with torch.no_grad():
#     for k in range(0, 5):
#         inputs, targets = dataiter.next()
#         outputs = model(inputs.to(device)).cpu()
#         inputs_ = inputs
#         plt.figure(figsize=(15, 7))
#         imshow(make_grid([inputs_[0], outputs[0], targets[0], (targets[0] - outputs[0]).abs()], normalize=True,
#                          value_range=(0, maxval[0])))
#         plt.axis('off')
#         img_input = inputs_[0].numpy()
#         img_noise = outputs[0].numpy()
#         img = targets[0].numpy()
#         print(k + 1)
#         print(' ssim:', ssim(img, img_noise, maxval[0].item()))
#         print('*ssim:', ssim(img, img_input, maxval[0].item()))
#         print(' psnr:', psnr(img, img_noise, maxval[0].item()))
#         print('*psnr:', psnr(img, img_input, maxval[0].item()))
#         print(' nmse:', nmse(img, img_noise, ))
#         print('*nmse:', nmse(img, img_input))
#         plt.savefig('output/imagenet_after_spt' + str(k) + '.png')
#         # plt.show()

#--------------------------------------------------------------------
# Test the model trained
print("*********** Testing ************", datetime.now())
#--------------------------------------------------------------------

"""Example reconstructions"""
valloader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=0,
                       generator=torch.Generator().manual_seed(1))
dataiter = iter(valloader)
model.eval()
maxval = torch.tensor([1.], device='cpu')

ssim_clean = []
ssim_noise = []
psnr_clean = []
psnr_noise = []
nmse_clean = []
nmse_noise = []

testing_data = 4000
with torch.no_grad():
    for k in range(0, testing_data):
        inputs, targets = dataiter.next()
        outputs = model(inputs.to(device)).cpu()
        inputs_ = inputs
        img_input = inputs_[0].numpy()
        img_noise = outputs[0].numpy()
        img = targets[0].numpy()

        if (k<5):
            plt.figure(figsize=(15, 7))
            imshow(make_grid([inputs_[0], outputs[0], targets[0], (targets[0] - outputs[0]).abs()], normalize=True,
                            value_range=(0, maxval[0])))
            plt.axis('off')
            plt.savefig('output/imagenet_after_spt' + str(k) + '.png')

        ssim_clean.append(ssim(img, img_noise, maxval[0].item()))
        ssim_noise.append(ssim(img, img_input, maxval[0].item()))
        psnr_clean.append(psnr(img, img_noise, maxval[0].item()))
        psnr_noise.append(psnr(img, img_input, maxval[0].item()))
        nmse_clean.append(nmse(img, img_noise, ))
        nmse_noise.append(nmse(img, img_input))

    output_stat = pd.DataFrame()
    output_stat['ssim_clean'] = ssim_clean
    output_stat['ssim_noise'] = ssim_noise
    output_stat['psnr_clean'] = psnr_clean
    output_stat['psnr_noise'] = psnr_noise
    output_stat['nmse_clean'] = nmse_clean
    output_stat['nmse_noise'] = nmse_noise
    output_stat.to_csv("output/breakdown.csv")

    # output_stat = pd.read_csv("output/testing_metrics.csv")
    testing_metrics = output_stat.mean(axis=0)
    output_stat = pd.concat([output_stat, testing_metrics])
    testing_metrics.to_csv("output/testing_metrics.csv")

    print(' ssim:', testing_metrics['ssim_clean'])
    print('*ssim:', testing_metrics['ssim_noise'])
    print(' psnr:', testing_metrics['psnr_clean'])
    print('*psnr:', testing_metrics['psnr_noise'])
    print(' nmse:', testing_metrics['nmse_clean'])
    print('*nmse:', testing_metrics['nmse_noise'])

print("*********** ENDED AT ************", datetime.now())

#--------------------------------------------------------------------
# Completed running a ViT network with Imagenet dataset
#--------------------------------------------------------------------
