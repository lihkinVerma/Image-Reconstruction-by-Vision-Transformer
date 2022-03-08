# pytorch
import torch
from torchvision.utils import make_grid
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# fastmri
import fastmri
from fastmri.data import subsample
from fastmri.data import transforms, mri_data
from fastmri.evaluate import ssim, psnr, nmse
from fastmri.losses import SSIMLoss
from fastmri.models import Unet

# other
from myutils import SSIM, PSNR
from mymodels import VisionTransformer, ReconNet
import os
import matplotlib.pyplot as plt
from myutils import imshow

# Device
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = 'cuda'

class fastMRIDataset(Dataset):
    def __init__(self, challenge, path, isval):
        """
        Dataloader for 4x acceleration and random sampling
        challenge: 'multicoil' or 'singlecoil'
        path: path to dataset
        isval: whether dataset is fastMRI's validation set or training set
        """
        self.challenge = challenge 
        self.data_path = path
        self.isval = isval

        self.data = mri_data.SliceDataset(
            root=self.data_path,
            transform=self.data_transform,
            challenge=self.challenge,
            use_dataset_cache=True,
            )

        self.mask_func = subsample.RandomMaskFunc( # RandomMaskFunc for knee, EquispacedMaskFunc for brain
            center_fractions=[0.08],
            accelerations=[4],
            )
            
    def data_transform(self, kspace, mask, target, data_attributes, filename, slice_num):
        if self.isval:
            seed = tuple(map(ord, filename))
        else:
            seed = None     
        kspace = transforms.to_tensor(kspace)
        masked_kspace, _ = transforms.apply_mask(kspace, self.mask_func, seed)        
        
        target = transforms.to_tensor(target)
        zero_fill = fastmri.ifft2c(masked_kspace)
        zero_fill = transforms.complex_center_crop(zero_fill, target.shape)   
        x = fastmri.complex_abs(zero_fill)
        
        if self.challenge == 'multicoil':
            x = fastmri.rss(x)

        x = x.unsqueeze(0)
        target = target.unsqueeze(0)
        
        return (x, target, data_attributes['max'])    

    def __len__(self,):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]

        return data
   
# Create dataset
challenge = 'multicoil' # 'multicoil' or 'singlecoil'
train_path = './traindata/' # path to fastmri's training data
val_path = './valdata/' # path to fastmri's validation data
dataset = fastMRIDataset(challenge=challenge, path=train_path, isval=False)
val_dataset = fastMRIDataset(challenge=challenge, path=val_path, isval=True)

ntrain = len(dataset) # number of training data
train_dataset, _ = torch.utils.data.random_split(dataset, [ntrain, len(dataset)-ntrain], generator=torch.Generator().manual_seed(42))
print(len(train_dataset))

batch_size = 1
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(42))
valloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

"""Init Model"""
## Vision Transformer
avrg_img_size = 340
patch_size = 10
depth = 10
num_heads = 16
embed_dim = 44

net = VisionTransformer(
    avrg_img_size=avrg_img_size, 
    patch_size=patch_size, 
    in_chans=1, embed_dim=embed_dim, 
    depth=depth, num_heads=num_heads,
    )

## Unet
# net = Unet(
#     in_chans=1,
#     out_chans=1,
#     chans=32,
#     num_pool_layers=4,
#     )

model = ReconNet(net).to(device)

print('#Params:', sum(p.numel() for p in model.parameters() if p.requires_grad))
print(model)

# Validate model
def validate(model):
    valloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)   
    model.eval()    
    ssim_ = SSIM().to(device)
    psnr_ = PSNR().to(device)
    psnrs = []
    ssims = []
    
    with torch.no_grad():
        for data in valloader:
            inputs, targets, maxval = data        
            outputs = model(inputs.to(device))
            ssims.append(ssim_(outputs, targets.to(device), maxval.to(device)))
            psnrs.append(psnr_(outputs, targets.to(device), maxval.to(device)))
    
    ssimval = torch.cat(ssims).mean()
    
    print(' Recon. PSNR: {:0.3f} pm {:0.2f}'.format(torch.cat(psnrs).mean(), 2*torch.cat(psnrs).std()))
    print(' Recon. SSIM: {:0.4f} pm {:0.3f}'.format(torch.cat(ssims).mean(), 2*torch.cat(ssims).std()))
                
    return (1-ssimval).item()

# Save model
def save_model(path, model, train_hist, val_hist, optimizer, scheduler=None):
    net = model.net
    if scheduler:
        checkpoint = {
            'model' :  ReconNet(net),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(), 
        }
    else:
        checkpoint = {
            'model' :  ReconNet(net),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        
    torch.save(train_hist, path + 'train_hist.pt')
    torch.save(val_hist, path + 'val_hist.pt')    
    torch.save(checkpoint,  path + 'checkpoint.pth')
    
"""Optimizer"""
criterion = SSIMLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0)
train_hist = []
val_hist = []
best_val = float("inf")
path = './' # Path for saving model checkpoint and loss history
num_epochs = 40
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.0003,
                                          total_steps=num_epochs, pct_start=0.1,
                                          anneal_strategy='linear',
                                          cycle_momentum=False,
                                          base_momentum=0., max_momentum=0., div_factor=0.1*num_epochs, final_div_factor=9)

"""Train Model"""
for epoch in range(0, num_epochs):
    model.train()
    train_loss = 0.0

    for data in trainloader:
        inputs, targets, maxval = data
        optimizer.zero_grad()
        outputs = model(inputs.to(device))
        loss = criterion(outputs, targets.to(device), maxval.to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=1.)
        optimizer.step()

        train_loss += loss.item()
    
    if scheduler:
        scheduler.step()
        
    train_hist.append(train_loss/len(trainloader))
    print('Epoch {}, Train loss.: {:0.4f}'.format(epoch+1, train_hist[-1]))
    
    if (epoch+1)%5==0:
        print('Validation:')
        val_hist.append(validate(model))        
        if val_hist[-1] < best_val:
            save_model(path, model, train_hist, val_hist, optimizer, scheduler=scheduler)
            best_val = val_hist[-1]


"""Loss History Plot"""
plt.plot(range(1,len(train_hist)+1), train_hist, 'r+-')
plt.plot(torch.linspace(5, len(train_hist), int(len(train_hist)/5)), val_hist, 'r*-')
plt.grid('on')

"""Example reconstructions"""
valloader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(0))
dataiter = iter(valloader)
model.eval()

with torch.no_grad():
    for k in range(0,5):    
        inputs, targets, maxval = dataiter.next()
        outputs = model(inputs.to(device)).cpu()
        plt.figure(figsize=(15,7))
        imshow(make_grid([inputs[0], outputs[0], targets[0], (targets[0]-outputs[0]).abs()],normalize = True, value_range=(0,maxval[0]/1.5)))
        plt.axis('off')
        img_input = inputs[0].numpy()
        img_recon = outputs[0].numpy()
        img = targets[0].numpy()
        print(k+1)
        print(' ssim:', ssim(img, img_recon, maxval[0].item()))
        print('*ssim:', ssim(img, img_input, maxval[0].item()))
        print(' psnr:', psnr(img, img_recon, maxval[0].item()))
        print('*psnr:', psnr(img, img_input, maxval[0].item()))
        print(' nmse:', nmse(img, img_recon, ))
        print('*nmse:', nmse(img, img_input))
        plt.savefig('outputs/fastmri'+k+'.png')
        # plt.show()