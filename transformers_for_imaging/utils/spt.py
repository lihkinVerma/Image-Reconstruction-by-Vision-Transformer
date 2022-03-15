import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange
import math

#This is created by Deepkamal
class ShiftedPatchTokenization(nn.Module):
    def __init__(self, in_dim, dim, merging_size=2, exist_class_t=False, is_pe=False): # added patch_size
        super().__init__()

        self.exist_class_t = exist_class_t

        if exist_class_t:
            self.class_linear = nn.Linear(in_dim, dim)

        self.is_pe = is_pe

        patch_dim = (in_dim * 5) * (merging_size[0] ** 2) # 20->500
        # patch_dim=18*18
        self.patch_shifting = PatchShifting(merging_size[0])
        self.merging = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=merging_size[0], p2=merging_size[0]),
            # nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim)
        )



    def forward(self, x):

        if self.exist_class_t:
            visual_tokens, class_token = x[:, 1:], x[:, (0,)]
            reshaped = rearrange(visual_tokens, 'b (h w) d -> b d h w', h=int(math.sqrt(x.size(1))))
            out_visual = self.patch_shifting(reshaped)
            out_visual = self.merging(out_visual)
            out_class = self.class_linear(class_token)
            out = torch.cat([out_class, out_visual], dim=1)

        else:
            out = x if self.is_pe else rearrange(x, 'b (h w) d -> b d h w', h=int(math.sqrt(x.size(1)))) # [64,1,180,180]
            out = self.patch_shifting(out) # [64, 5, 180, 180]
            out = self.merging(out) # [4, 784, 704] [4, 704, 28, 28]
            out = out.view(out.shape[0], out.shape[2], int(math.sqrt(out.shape[1])), int(math.sqrt(out.shape[1])))
        return out


class PatchShifting(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        if isinstance(patch_size, tuple):
            self.shift = int(patch_size[0] * (1 / 2))
        else:
            self.shift = int(patch_size * (1 / 2))

    def forward(self, x):
        x_pad = torch.nn.functional.pad(x, (self.shift, self.shift, self.shift, self.shift))
        # if self.is_mean:
        #     x_pad = x_pad.mean(dim=1, keepdim = True)

        """ 4 cardinal directions """
        #############################
        # x_l2 = x_pad[:, :, self.shift:-self.shift, :-self.shift*2]
        # x_r2 = x_pad[:, :, self.shift:-self.shift, self.shift*2:]
        # x_t2 = x_pad[:, :, :-self.shift*2, self.shift:-self.shift]
        # x_b2 = x_pad[:, :, self.shift*2:, self.shift:-self.shift]
        # x_cat = torch.cat([x, x_l2, x_r2, x_t2, x_b2], dim=1)
        #############################

        """ 4 diagonal directions """
        # #############################
        x_lu = x_pad[:, :, :-self.shift * 2, :-self.shift * 2]
        x_ru = x_pad[:, :, :-self.shift * 2, self.shift * 2:]
        x_lb = x_pad[:, :, self.shift * 2:, :-self.shift * 2]
        x_rb = x_pad[:, :, self.shift * 2:, self.shift * 2:]
        x_cat = torch.cat([x, x_lu, x_ru, x_lb, x_rb], dim=1)
        # #############################

        """ 8 cardinal directions """
        #############################
        # x_l2 = x_pad[:, :, self.shift:-self.shift, :-self.shift*2]
        # x_r2 = x_pad[:, :, self.shift:-self.shift, self.shift*2:]
        # x_t2 = x_pad[:, :, :-self.shift*2, self.shift:-self.shift]
        # x_b2 = x_pad[:, :, self.shift*2:, self.shift:-self.shift]
        # x_lu = x_pad[:, :, :-self.shift*2, :-self.shift*2]
        # x_ru = x_pad[:, :, :-self.shift*2, self.shift*2:]
        # x_lb = x_pad[:, :, self.shift*2:, :-self.shift*2]
        # x_rb = x_pad[:, :, self.shift*2:, self.shift*2:]
        # x_cat = torch.cat([x, x_l2, x_r2, x_t2, x_b2, x_lu, x_ru, x_lb, x_rb], dim=1)
        #############################

        # out = self.out(x_cat)
        out = x_cat

        return out
